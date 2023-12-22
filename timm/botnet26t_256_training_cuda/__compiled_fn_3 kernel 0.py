
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


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyej23vyw5763kmux2dywgfrfqsdpwbf6bflpsog6h63cyycpqw.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 72
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


# kernel path: /tmp/torchinductor_youkaichao/3o/c3o65mbemxcx7wqsyjywjhgnft4qwjiyoqgf4lcvaeqofhwnd6bv.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (216*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu35sx4ulkrbt5hxuk2tonj5rj3yaij3mcrqim6khmk4xlfnyt3o.py
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
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
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


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxxn5napgbjqhvjn2f4dcq56ej6ulkhuyatecs2tyodzqc2sc7t.py
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
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2c/c2coczahrpj4wynkr6bnhwkbrkarvdhudyw232w6dzxuxc3et3nx.py
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


# kernel path: /tmp/torchinductor_youkaichao/zo/czo4ye555ijreyldlwjxr46d5xbmsdlk3ao6zzc3dokazwroftdb.py
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
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/h7/ch7ljnuhqct45dygf2onapox7rhgebtei7ecqtezlhefzoj73tdx.py
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
    ynumel = 192
    xnumel = 16384
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (393216*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cag5c4z3piltlxwmkz74warlabxxykbdervd26vpfauxcxuinbhk.py
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
    xnumel = 24576
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r2) + (3072*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7e/c7ejmb5qpn2xd27qyacegktg2h3pseymaseri55vu3nudk3vbftc.py
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
    xnumel = 192
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
        tmp0 = tl.load(in_ptr0 + (x1 + (24*r2) + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (24*r2) + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (24*r2) + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (24*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (24*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (24*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpspy24z3kyfobobfanxeigvrvo7xpnqomb3s3laiy6ptyz4opju.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (24*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/z5/cz532su2bvby4no4yycbzpo5weacf7ow77q577pm2jpjv4mngr2b.py
# Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_4 => relu
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
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
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


# kernel path: /tmp/torchinductor_youkaichao/hr/chrxf63svznypcpawl7d74zabuarn6nwcdxha224wr6d3rma3wdy.py
# Source Nodes: [x_5], Original ATen: [aten.convolution]
# x_5 => convolution_1
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdszv57b7ms4dwe2foyz4e2ck55p2yss4vuikgxarydj3zqsgcz.py
# Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
# x_6 => var_mean_1
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


# kernel path: /tmp/torchinductor_youkaichao/nl/cnlqruhohdomvkjf7pu7t3jlppvmrnbmgwmuipunymn6oqvm7tzd.py
# Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
# x_6 => var_mean_1
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


# kernel path: /tmp/torchinductor_youkaichao/nc/cncqtwgfdu2fqfxrphymb2mlcqirkx4mw3w2qnjq5edd73qgdvlr.py
# Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
# x_6 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdziwov6ah7djuwovu34rxeb44zg4hsr2xjdt2g5lg2amcrrmut.py
# Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_6 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
# x_9 => relu_1
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


# kernel path: /tmp/torchinductor_youkaichao/hh/chhvdwodsk3u4ngrx6ajtmgbaitlymhs5dbqnyf7t2ymn4mxxwsx.py
# Source Nodes: [x_10], Original ATen: [aten.convolution]
# x_10 => convolution_2
triton_poi_fused_convolution_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (1048576*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3uymokvopspsybjyilkxxvx65j6reskk57vkotptidoylul7nn.py
# Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
# x_11 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzeskk5sbijoqixwjiqnphcxbwkavagpln6koabnu3h3cpl7v62.py
# Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
# x_11 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/23/c23za5jyn3pkt55oaunsm2zizsmed46xihccsjebwapaqa4psvdr.py
# Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
# x_11 => add_11, add_12, add_13, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/wc/cwccu24zlabebbefq2c2zi24jy25ofdd6msw34vrz3izixuezwbw.py
# Source Nodes: [x_11, x_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_11 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# x_14 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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


# kernel path: /tmp/torchinductor_youkaichao/63/c63gi535jnbrdlaqmdyaowfoy24kfjapqf6ok5nuhzlzszs5nhlw.py
# Source Nodes: [shortcut], Original ATen: [aten.max_pool2d_with_indices]
# shortcut => getitem_6, getitem_7
triton_poi_fused_max_pool2d_with_indices_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4096) % 64
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x5 = (xindex // 4096)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-8256) + x0 + (128*x1) + (16384*x5)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-8192) + x0 + (128*x1) + (16384*x5)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x1)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-8128) + x0 + (128*x1) + (16384*x5)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-64) + x0 + (128*x1) + (16384*x5)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x0 + (128*x1) + (16384*x5)), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (16384*x5)), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x2)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (8128 + x0 + (128*x1) + (16384*x5)), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (8192 + x0 + (128*x1) + (16384*x5)), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (8256 + x0 + (128*x1) + (16384*x5)), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = (-128) + (2*x1) + (256*x2)
    tmp72 = (-129) + (2*x1) + (256*x2)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = (-127) + (2*x1) + (256*x2)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = (-1) + (2*x1) + (256*x2)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = (2*x1) + (256*x2)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 1 + (2*x1) + (256*x2)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 127 + (2*x1) + (256*x2)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 128 + (2*x1) + (256*x2)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 129 + (2*x1) + (256*x2)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x6), tmp69, None)
    tl.store(out_ptr1 + (x6), tmp94, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77vy3ee3bp5n7ksp2l3lzdc55dgk4im535jo4hs7fj63crmyucf.py
# Source Nodes: [x_16], Original ATen: [aten.convolution]
# x_16 => convolution_3
triton_poi_fused_convolution_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5k2x6fkv57nlvio7a2zmhxd5zolryxoch2n243w3c5tdxohzal.py
# Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
# x_17 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xp/cxpxeyrgztbx2hpi6anxcn36qkuicu7xv6hhx7me45nsdnme7kqu.py
# Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
# x_17 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6jige5kzy7cja3ncqs6d2ohj4u5ecipsw4gzvkdbkb376qyjib.py
# Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
# x_17 => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/gl/cglbnojximm4blgpxtri5iujhkaeuwzfbwjh7y36fovu3s26qqhw.py
# Source Nodes: [x_17, x_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_17 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_21 => relu_3
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
    xnumel = 2097152
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


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7ltn66jv5rfnapihnadrphqgjifaz4owqfozlfhg7m4rykacjr.py
# Source Nodes: [x_30], Original ATen: [aten.convolution]
# x_30 => convolution_5
triton_poi_fused_convolution_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (1048576*y1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjpoganohwdtjdiiroahgmwpjlued4tqg7wuwsf4eudjb4r6detw.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
# x_31 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5fizbvju22vbgduy34vmplbcs7nzpijd7i3kyvnsvzysdumnme.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
# x_31 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6n/c6nuztdkehpaex6c4sxpfwzw4u6qmb3soklkjzbj4hkx4jalmpu2.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
# x_31 => add_26, add_27, add_28, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3bo2hktxdt2bph3vyuin3ofa34uslhdd3piofuxghkm3lrpe5c.py
# Source Nodes: [shortcut_1, x_31, x_39, x_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_1 => relu_5
# x_31 => add_26, add_29, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
# x_39 => add_31, add_34, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# x_43 => add_35
triton_poi_fused__native_batch_norm_legit_functional_add_relu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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


# kernel path: /tmp/torchinductor_youkaichao/jj/cjj5ftl3rvqyngvxw5kxmpaqxk3e7eypqihzsvmfjypnpxiygb6i.py
# Source Nodes: [shortcut_2, x_59, x_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_2 => relu_8
# x_59 => add_47, add_50, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
# x_66 => add_51
triton_poi_fused__native_batch_norm_legit_functional_add_relu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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
    tmp4 = 32768.0
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


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxf54cc3zhskblsdtuhbc6md7gwybsmbncua6kzdcjzm2glhtj4.py
# Source Nodes: [x_67], Original ATen: [aten.convolution]
# x_67 => convolution_10
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gx/cgxfc7wp4dzawca42cjvcwww56wnxewr4vettxvt3lncoatmmhep.py
# Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
# x_68 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5w7qncddkqp2xtdrne5nfh6fxvjkqijiddpbahgwl5gd2uqke6.py
# Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
# x_68 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vd/cvd7vaekroi2sccfs26zqx2a56a4tvntlwct3rgp2p3xrgbr3mwf.py
# Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
# x_68 => add_53, add_54, add_55, mul_71, mul_72, mul_73, mul_74, mul_75, rsqrt_10, squeeze_31, var_mean_10
triton_per_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3rq3jks2ma5jqzyk7suhsylw3zny2fdvdwpsfrbakphe54gskt.py
# Source Nodes: [x_68, x_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_68 => add_53, add_56, mul_70, mul_76, rsqrt_10, sub_10, var_mean_10
# x_72 => relu_9
triton_poi_fused__native_batch_norm_legit_functional_relu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_38', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pz3e4d5ozq74xuwttwbtyxt53acqms6xx7kqe3cwhdzltyumhd.py
# Source Nodes: [x_73], Original ATen: [aten.convolution]
# x_73 => convolution_11
triton_poi_fused_convolution_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask)
    tl.store(out_ptr0 + (y0 + (128*x2) + (131072*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gs/cgsysa3qsslezhqfad7n6xxl5waj4tpakwemr2meusz65d6csw44.py
# Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
# x_74 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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


# kernel path: /tmp/torchinductor_youkaichao/no/cnokx4hiedv5c4edpeohbp27xca75lfeuiztzdkkriggrfnty4x5.py
# Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
# x_74 => add_58, add_59, add_60, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_41', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5thsca5fznrforggvmw6nqkk537s7dloic5ygvzgm4xnpkjplc.py
# Source Nodes: [x_74, x_78], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_74 => add_58, add_61, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# x_78 => relu_10
triton_poi_fused__native_batch_norm_legit_functional_relu_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /tmp/torchinductor_youkaichao/rm/crmkju4wobwqovzdevmxpqr5kxp6vfalumancg5yvdmd33tp3mya.py
# Source Nodes: [x_81], Original ATen: [aten.convolution]
# x_81 => convolution_12
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
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (524288*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ud/cudhd2fk6pi7t5kzm763uy5xpucanmqy23laibvbstinutkujsgb.py
# Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
# x_82 => var_mean_12
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
    xnumel = 32768
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5u/c5utx5ujisz2e64cnjvn6355dzf4wqqmb74u4aw3b3lqq6epybju.py
# Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
# x_82 => add_63, add_64, add_65, mul_85, mul_86, mul_87, mul_88, mul_89, rsqrt_12, squeeze_37, var_mean_12
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/sq/csq5htezktoyuoqk4b7iwm2j2g4okrk3rur5z2jujwjdnwkiw2tj.py
# Source Nodes: [shortcut_3, x_82, x_90, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_3 => relu_11
# x_82 => add_63, add_66, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
# x_90 => add_68, add_71, mul_91, mul_97, rsqrt_13, sub_13, var_mean_13
# x_94 => add_72
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
    xnumel = 4194304
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


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6r44dyectaqsi2kxqdx6ssti5nlemgasnjdt3us3xph3pwo2f5.py
# Source Nodes: [shortcut_4, x_110, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_4 => relu_14
# x_110 => add_84, add_87, mul_112, mul_118, rsqrt_16, sub_16, var_mean_16
# x_117 => add_88
triton_poi_fused__native_batch_norm_legit_functional_add_relu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5hfa32pb54b645sexxyv6kxtwuvqwv7euy45eiha5oa63wnzrx.py
# Source Nodes: [x_118], Original ATen: [aten.convolution]
# x_118 => convolution_17
triton_poi_fused_convolution_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (262144*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wo/cwoogvozdd7bq6mvlczz3lzrrhanu37cb6xobs65c2qafex2t55q.py
# Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
# x_119 => var_mean_17
triton_red_fused__native_batch_norm_legit_functional_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlrmopfjpbmfymr2ccj4qjmbnh4soppoucxxus52a6n3zfrhcmu.py
# Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
# x_119 => add_90, add_91, add_92, mul_120, mul_121, mul_122, mul_123, mul_124, rsqrt_17, squeeze_52, var_mean_17
triton_per_fused__native_batch_norm_legit_functional_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_50', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqba3gitc3iuk62zxm7t22nemepxiflahskkcdjqhhlkxoxiqcq.py
# Source Nodes: [x_119, x_123], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_119 => add_90, add_93, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
# x_123 => relu_15
triton_poi_fused__native_batch_norm_legit_functional_relu_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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


# kernel path: /tmp/torchinductor_youkaichao/ze/czeidcdjhq6mkielenp6wbihpbm23g3ko7yoapibeuffkfwjsff2.py
# Source Nodes: [x_124], Original ATen: [aten.convolution]
# x_124 => convolution_18
triton_poi_fused_convolution_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (256*x2) + (65536*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yi/cyigaegoauvjvyq2yum7ynd4r4s3zf7iai2oxwq5f6uhmfss6kyf.py
# Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
# x_125 => var_mean_18
triton_red_fused__native_batch_norm_legit_functional_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nk/cnkyicyglq3opaypoge4xd3zxc25pj5w2evj24jbv4sibcvfqvhr.py
# Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
# x_125 => add_95, add_96, add_97, mul_127, mul_128, mul_129, mul_130, mul_131, rsqrt_18, squeeze_55, var_mean_18
triton_per_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_54', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/mz/cmz3phds36yeyxrh3vkeljvd2no5nbcqc7elggbj4lvtyu3ongev.py
# Source Nodes: [x_125, x_129], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_125 => add_95, add_98, mul_126, mul_132, rsqrt_18, sub_18, var_mean_18
# x_129 => relu_16
triton_poi_fused__native_batch_norm_legit_functional_relu_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: /tmp/torchinductor_youkaichao/vr/cvr3vq6ropgmp45giwc4ueonnwc2ixx5amyrpqqo55uy7iaxyjgj.py
# Source Nodes: [x_132], Original ATen: [aten.convolution]
# x_132 => convolution_19
triton_poi_fused_convolution_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (1024*x2) + (262144*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ak/cakwuuf5byqiot25qwzegpcsao343r24ejmujkuutsxvzqhscm25.py
# Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
# x_133 => var_mean_19
triton_red_fused__native_batch_norm_legit_functional_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/pc/cpc3bqvxydrjfxasmb7ipypgwglshn6jf57vvu52tak5zk33nupu.py
# Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
# x_133 => add_100, add_101, add_102, mul_134, mul_135, mul_136, mul_137, mul_138, rsqrt_19, squeeze_58, var_mean_19
triton_per_fused__native_batch_norm_legit_functional_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_58', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/jy/cjyyxpbzltvrndwj5vjhsfdunm2uxtsxk22ne6aybd5mahejhuq6.py
# Source Nodes: [shortcut_5, x_133, x_141, x_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_5 => relu_17
# x_133 => add_100, add_103, mul_133, mul_139, rsqrt_19, sub_19, var_mean_19
# x_141 => add_105, add_108, mul_140, mul_146, rsqrt_20, sub_20, var_mean_20
# x_145 => add_109
triton_poi_fused__native_batch_norm_legit_functional_add_relu_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_59', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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


# kernel path: /tmp/torchinductor_youkaichao/3w/c3w6pbxjgzrbj4zbw3bhcgyjv7vrpz3wmlzogl776iufi6vx6oq3.py
# Source Nodes: [reshape], Original ATen: [aten.clone]
# reshape => clone
triton_poi_fused_clone_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 65536
    x1 = (xindex // 65536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196608*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvl5qjg7le6yfwtnpr7rpzuucj4w6oz3llt33swpc5dfdzngsit.py
# Source Nodes: [k_1], Original ATen: [aten.clone]
# k_1 => clone_1
triton_poi_fused_clone_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 65536
    x1 = (xindex // 65536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (65536 + x0 + (196608*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7ubaqjdpawwb7er2exbkp7lb72bpmr5urk7icd5x5zweljwej6z.py
# Source Nodes: [x_154], Original ATen: [aten._unsafe_view, aten.clone]
# x_154 => clone_3, view_7
triton_poi_fused__unsafe_view_clone_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((16*((((16*((y0 // 16) % 16)) + (y0 % 16)) // 16) % 16)) + (256*((((16*((y0 // 16) % 16)) + (256*x1) + (16384*(y0 // 256)) + (y0 % 16)) // 256) % 2048)) + (y0 % 16)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (64*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hqgr6dgjlkvgpnb7lcxtlzebpgmwoufiqwkv6nuwgyaxdquyfk.py
# Source Nodes: [x_158], Original ATen: [aten._unsafe_view, aten.clone]
# x_158 => clone_4, view_13
triton_poi_fused__unsafe_view_clone_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((16*((((16*(x1 % 16)) + ((x1 // 16) % 16)) // 16) % 16)) + (256*((((16*(x1 % 16)) + (256*x0) + (16384*(x1 // 256)) + ((x1 // 16) % 16)) // 256) % 2048)) + ((x1 // 16) % 16)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2jqemybznmktiufdaxm5jjziuqxpofnhfyl53oqpctb4ysrsai.py
# Source Nodes: [attn, attn_1, mul], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn => add_116
# attn_1 => amax, div, exp, sub_22, sum_1
# mul => mul_154
triton_red_fused__softmax_add_mul_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.125
        tmp2 = tmp0 * tmp1
        tmp3 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp4 = tl.full([1, 1], 512, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp7 = tl.full([1, 1], 31, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp5, tmp12, tmp13)
        tmp15 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp16 = tmp15 < tmp4
        tmp17 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp18 = tmp17 < tmp7
        tmp19 = tmp18 & tmp16
        tmp20 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp16, tmp22, tmp23)
        tmp25 = tmp14 + tmp24
        tmp26 = tmp2 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp30 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = 0.125
        tmp32 = tmp30 * tmp31
        tmp33 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp34 = tl.full([1, 1], 512, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp37 = tl.full([1, 1], 31, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp39, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
        tmp42 = tl.where(tmp39, tmp40, tmp41)
        tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
        tmp44 = tl.where(tmp35, tmp42, tmp43)
        tmp45 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp46 = tmp45 < tmp34
        tmp47 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp48 = tmp47 < tmp37
        tmp49 = tmp48 & tmp46
        tmp50 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp49, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
        tmp52 = tl.where(tmp49, tmp50, tmp51)
        tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
        tmp54 = tl.where(tmp46, tmp52, tmp53)
        tmp55 = tmp44 + tmp54
        tmp56 = tmp32 + tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(rmask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp62 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = 0.125
        tmp64 = tmp62 * tmp63
        tmp65 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp66 = tl.full([1, 1], 512, tl.int64)
        tmp67 = tmp65 < tmp66
        tmp68 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp69 = tl.full([1, 1], 31, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tmp70 & tmp67
        tmp72 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp71, tmp72, tmp73)
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp67, tmp74, tmp75)
        tmp77 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp78 = tmp77 < tmp66
        tmp79 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp80 = tmp79 < tmp69
        tmp81 = tmp80 & tmp78
        tmp82 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
        tmp84 = tl.where(tmp81, tmp82, tmp83)
        tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
        tmp86 = tl.where(tmp78, tmp84, tmp85)
        tmp87 = tmp76 + tmp86
        tmp88 = tmp64 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl.exp(tmp89)
        tmp91 = tmp90 / tmp60
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp91, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zr/czrqhciyarvpnyl65geod5f2iz2k5ynxtvrsyimxlqrec7zzqpqi.py
# Source Nodes: [reshape_2], Original ATen: [aten.clone]
# reshape_2 => clone_2
triton_poi_fused_clone_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 65536
    x1 = (xindex // 65536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (131072 + x0 + (196608*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caeclpm5n2u3wh4ex77uvxboct6lm2wz6ycskv3bvxpaf3xwhpk7.py
# Source Nodes: [x_163], Original ATen: [aten._native_batch_norm_legit_functional]
# x_163 => var_mean_22
triton_red_fused__native_batch_norm_legit_functional_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        tmp0 = tl.load(in_ptr0 + ((64*(((16*(((r2 + (128*x1)) // 16) % 16)) + (r2 % 16)) % 256)) + (16384*((((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (65536*((r2 + (128*x1)) // 256)) + (r2 % 16)) // 16384) % 32)) + ((((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (r2 % 16)) // 256) % 64)), rmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/os/costxhljxmvwjopdonlwoekrg67xet7ky7kr5eicf7xbyljpw535.py
# Source Nodes: [x_163, x_166], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_163 => add_118, add_121, mul_155, mul_161, rsqrt_22, sub_23, var_mean_22
# x_166 => relu_19
triton_poi_fused__native_batch_norm_legit_functional_relu_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 256
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (16384*((x1 + (256*x0)) // 16384)) + (65536*x2) + (x0 % 64)), None)
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
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/at/catenn23uu4zri3ts55ojvvjq66oe7gv2axcn5grwgmrr7i5pt3e.py
# Source Nodes: [shortcut_6, x_168, x_174], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_6 => relu_20
# x_168 => add_123, add_126, mul_162, mul_168, rsqrt_23, sub_24, var_mean_23
# x_174 => add_127
triton_poi_fused__native_batch_norm_legit_functional_add_relu_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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


# kernel path: /tmp/torchinductor_youkaichao/44/c446bd7mze6wwxqbjbo6dhb7tculbhq2cgx45opa3qbhsvb7nhbo.py
# Source Nodes: [x_175], Original ATen: [aten.convolution]
# x_175 => convolution_24
triton_poi_fused_convolution_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (131072*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/ceczw4d26ul66winytd7ok3zvgbtqepqlkkcphsh7jepsa2ke5zf.py
# Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_functional]
# x_176 => var_mean_24
triton_red_fused__native_batch_norm_legit_functional_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/26/c26kcnrjrduqz6gxg3wbrmzgbrm7r6zxhxovaglvuqwx4sgdnvi4.py
# Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_functional]
# x_176 => add_129, add_130, add_131, mul_170, mul_171, mul_172, mul_173, mul_174, rsqrt_24, squeeze_73, var_mean_24
triton_per_fused__native_batch_norm_legit_functional_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_71', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/rf/crfxmlvowwbnvg3vjmtzt2ffltkv65cl4ehbjbtz26knfintdq57.py
# Source Nodes: [x_176, x_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_176 => add_129, add_132, mul_169, mul_175, rsqrt_24, sub_25, var_mean_24
# x_180 => relu_21
triton_poi_fused__native_batch_norm_legit_functional_relu_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /tmp/torchinductor_youkaichao/xk/cxka53l3lg2x6zu7bw65plrjnzipoehzuzde3iaf2m245xtskjhx.py
# Source Nodes: [reshape_12], Original ATen: [aten.clone]
# reshape_12 => clone_7
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 131072
    x1 = (xindex // 131072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (393216*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7f/c7fejnxleqkbx2rced2iju6ka7an2rvke6t36grftjndpg53idye.py
# Source Nodes: [k_3], Original ATen: [aten.clone]
# k_3 => clone_8
triton_poi_fused_clone_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 131072
    x1 = (xindex // 131072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (131072 + x0 + (393216*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/df/cdfyg3vopmo7ry2opcqyocuwds3auv37slajnfg3ntsuw65mp57y.py
# Source Nodes: [x_183], Original ATen: [aten._unsafe_view, aten.clone]
# x_183 => clone_10, view_31
triton_poi_fused__unsafe_view_clone_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((16*((((16*((y0 // 16) % 16)) + (y0 % 16)) // 16) % 16)) + (256*((((16*((y0 // 16) % 16)) + (256*x1) + (32768*(y0 // 256)) + (y0 % 16)) // 256) % 4096)) + (y0 % 16)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (128*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ad/cad4hv4h7kfdzhjwuppqzyusx6kshgpwg7p44jcmd66fsnywayu3.py
# Source Nodes: [x_187], Original ATen: [aten._unsafe_view, aten.clone]
# x_187 => clone_11, view_37
triton_poi_fused__unsafe_view_clone_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((16*((((16*(x1 % 16)) + ((x1 // 16) % 16)) // 16) % 16)) + (256*((((16*(x1 % 16)) + (256*x0) + (32768*(x1 // 256)) + ((x1 // 16) % 16)) // 256) % 4096)) + ((x1 // 16) % 16)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yr/cyrftub2buxeawth7xp6uhhhdivolt6qxxon5z5kuhdyaxbfdgrj.py
# Source Nodes: [attn_2, attn_3, mul_1], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_2 => add_134
# attn_3 => amax_1, div_1, exp_1, sub_26, sum_2
# mul_1 => mul_176
triton_red_fused__softmax_add_mul_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.08838834764831845
        tmp2 = tmp0 * tmp1
        tmp3 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp4 = tl.full([1, 1], 512, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp7 = tl.full([1, 1], 31, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp5, tmp12, tmp13)
        tmp15 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp16 = tmp15 < tmp4
        tmp17 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp18 = tmp17 < tmp7
        tmp19 = tmp18 & tmp16
        tmp20 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp16, tmp22, tmp23)
        tmp25 = tmp14 + tmp24
        tmp26 = tmp2 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp30 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = 0.08838834764831845
        tmp32 = tmp30 * tmp31
        tmp33 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp34 = tl.full([1, 1], 512, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp37 = tl.full([1, 1], 31, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp39, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
        tmp42 = tl.where(tmp39, tmp40, tmp41)
        tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
        tmp44 = tl.where(tmp35, tmp42, tmp43)
        tmp45 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp46 = tmp45 < tmp34
        tmp47 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp48 = tmp47 < tmp37
        tmp49 = tmp48 & tmp46
        tmp50 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp49, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
        tmp52 = tl.where(tmp49, tmp50, tmp51)
        tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
        tmp54 = tl.where(tmp46, tmp52, tmp53)
        tmp55 = tmp44 + tmp54
        tmp56 = tmp32 + tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(rmask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp62 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = 0.08838834764831845
        tmp64 = tmp62 * tmp63
        tmp65 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp66 = tl.full([1, 1], 512, tl.int64)
        tmp67 = tmp65 < tmp66
        tmp68 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp69 = tl.full([1, 1], 31, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tmp70 & tmp67
        tmp72 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp71, tmp72, tmp73)
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp67, tmp74, tmp75)
        tmp77 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp78 = tmp77 < tmp66
        tmp79 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp80 = tmp79 < tmp69
        tmp81 = tmp80 & tmp78
        tmp82 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
        tmp84 = tl.where(tmp81, tmp82, tmp83)
        tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
        tmp86 = tl.where(tmp78, tmp84, tmp85)
        tmp87 = tmp76 + tmp86
        tmp88 = tmp64 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl.exp(tmp89)
        tmp91 = tmp90 / tmp60
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp91, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwym2pkm45jgrpdwsorawrsxnav6nibyg7uk4syoh4ppbxq7r2e3.py
# Source Nodes: [reshape_14], Original ATen: [aten.clone]
# reshape_14 => clone_9
triton_poi_fused_clone_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 131072
    x1 = (xindex // 131072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (262144 + x0 + (393216*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bp/cbposchzgzva3fomtd4lbpswpbf6cinenwozyjuvxfcjqjmgca5d.py
# Source Nodes: [out_2], Original ATen: [aten._unsafe_view, aten.clone]
# out_2 => clone_13, view_47
triton_poi_fused__unsafe_view_clone_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 256
    x2 = (xindex // 131072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*x1) + (32768*((x1 + (256*x0)) // 32768)) + (131072*x2) + (x0 % 128)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rl/crl35h5vwte4a5vx5qumbym5a2qqamgumupahwuzznzdl6f6jxsn.py
# Source Nodes: [x_191], Original ATen: [aten.avg_pool2d]
# x_191 => avg_pool2d
triton_poi_fused_avg_pool2d_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 8
    x2 = (xindex // 4096)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x1) + (16384*x2)), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (16384*x2)), None)
    tmp3 = tl.load(in_ptr0 + (8192 + x0 + (1024*x1) + (16384*x2)), None)
    tmp5 = tl.load(in_ptr0 + (8704 + x0 + (1024*x1) + (16384*x2)), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cyw4lnninjerhkz3rbawoi3np5nllhundkojjl47n6el4tc6xpf4.py
# Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
# x_192 => var_mean_25
triton_red_fused__native_batch_norm_legit_functional_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/36/c36clzr5llsh6qv3m5xcahfkcfgumyvxlkikdoyl6d3wvhqbvamc.py
# Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
# x_192 => add_136, add_137, add_138, mul_178, mul_179, mul_180, mul_181, mul_182, rsqrt_25, squeeze_76, var_mean_25
triton_per_fused__native_batch_norm_legit_functional_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_82', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/ju/cjuo7kczznuexlqxtzni55bfobgseefutwadbrzbq3yab2k25svf.py
# Source Nodes: [x_192, x_195], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_192 => add_136, add_139, mul_177, mul_183, rsqrt_25, sub_27, var_mean_25
# x_195 => relu_22
triton_poi_fused__native_batch_norm_legit_functional_relu_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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


# kernel path: /tmp/torchinductor_youkaichao/id/cidve4mrkqvhmg5dlwl736qgv2qulavmmnqx2xcgqyvd4gulygsi.py
# Source Nodes: [x_196], Original ATen: [aten.convolution]
# x_196 => convolution_26
triton_poi_fused_convolution_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (2048*x2) + (131072*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h4/ch4inhbd25yll3wsxtt2oes7j3pd7suey4fupsopkhh7m3dgsryk.py
# Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
# x_197 => var_mean_26
triton_red_fused__native_batch_norm_legit_functional_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxnroohxhckdhmcjfq72kb7ckzm2edd55dflbj3iyv4o47wnweu.py
# Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
# x_197 => add_141, add_142, add_143, mul_185, mul_186, mul_187, mul_188, mul_189, rsqrt_26, squeeze_79, var_mean_26
triton_per_fused__native_batch_norm_legit_functional_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tl.store(out_ptr2 + (x0), tmp20, None)
    tl.store(out_ptr4 + (x0), tmp26, None)
    tl.store(out_ptr6 + (x0), tmp32, None)
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(out_ptr1 + (x0), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zu/czus2nma3nhgb2q7btan6cdel256zlvmlzdvqcytb6xrl6zaliqb.py
# Source Nodes: [shortcut_7, x_197, x_204, x_208], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_7 => relu_23
# x_197 => add_141, add_144, mul_184, mul_190, rsqrt_26, sub_28, var_mean_26
# x_204 => add_146, add_149, mul_191, mul_197, rsqrt_27, sub_29, var_mean_27
# x_208 => add_150
triton_poi_fused__native_batch_norm_legit_functional_add_relu_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_87', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmezvczfeb7m6z6m3e7sbm6ib6z7ktwkhuopcv2vndienhe4du5.py
# Source Nodes: [x_209], Original ATen: [aten.convolution]
# x_209 => convolution_28
triton_poi_fused_convolution_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (32768*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jg/cjgmhjagwetx7ceakiwkpopxxauszpdxwk7b2xsvc2gibezubwqh.py
# Source Nodes: [reshape_24], Original ATen: [aten.clone]
# reshape_24 => clone_14
triton_poi_fused_clone_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = (xindex // 32768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (98304*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7h/c7hdliyioxssealwqvvh674he3nkupfm6qezlifu2ovbejhplkir.py
# Source Nodes: [k_5], Original ATen: [aten.clone]
# k_5 => clone_15
triton_poi_fused_clone_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = (xindex // 32768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (32768 + x0 + (98304*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fp/cfpznwy62ldxwz3qpnitxj2buhov6q4l3ubl5k52zxuhls5wirju.py
# Source Nodes: [x_217], Original ATen: [aten._unsafe_view, aten.clone]
# x_217 => clone_17, view_55
triton_poi_fused__unsafe_view_clone_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((8*((((8*((y0 // 8) % 8)) + (y0 % 8)) // 8) % 8)) + (64*((((8*((y0 // 8) % 8)) + (64*x1) + (8192*(y0 // 64)) + (y0 % 8)) // 64) % 4096)) + (y0 % 8)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (128*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6n73zeuwimsdtkqgl23wc55q4kamaeksmhsv472q27zk72ohtxu.py
# Source Nodes: [x_221], Original ATen: [aten._unsafe_view, aten.clone]
# x_221 => clone_18, view_61
triton_poi_fused__unsafe_view_clone_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((8*((((8*(x1 % 8)) + ((x1 // 8) % 8)) // 8) % 8)) + (64*((((8*(x1 % 8)) + (64*x0) + (8192*(x1 // 64)) + ((x1 // 8) % 8)) // 64) % 4096)) + ((x1 // 8) % 8)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3m/c3mzd5qg52o6qsnbrlgxrlrvgpvpkzyumgmof6uqnm4m6b3v2oa5.py
# Source Nodes: [attn_4, attn_5, mul_2], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_4 => add_157
# attn_5 => amax_2, div_2, exp_2, sub_31, sum_3
# mul_2 => mul_205
triton_per_fused__softmax_add_mul_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = 0.08838834764831845
    tmp2 = tmp0 * tmp1
    tmp3 = 7 + (15*(x0 // 8)) + (r2 // 8)
    tmp4 = tl.full([1, 1], 128, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = (7 + (15*(x0 // 8)) + (r2 // 8)) % 16
    tmp7 = tl.full([1, 1], 15, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + ((15*((7 + (15*(x0 // 8)) + (r2 // 8)) // 16)) + (120*(x0 % 8)) + (960*x1) + ((7 + (15*(x0 // 8)) + (r2 // 8)) % 16)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = 7 + (15*(x0 % 8)) + (r2 % 8)
    tmp16 = tmp15 < tmp4
    tmp17 = (7 + (15*(x0 % 8)) + (r2 % 8)) % 16
    tmp18 = tmp17 < tmp7
    tmp19 = tmp18 & tmp16
    tmp20 = tl.load(in_ptr2 + ((15*(((7 + (15*(x0 % 8)) + (r2 % 8)) // 16) % 8)) + (120*(x0 // 8)) + (960*x1) + ((7 + (15*(x0 % 8)) + (r2 % 8)) % 16)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp16, tmp22, tmp23)
    tmp25 = tmp14 + tmp24
    tmp26 = tmp2 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(rmask, tmp27, float("-inf"))
    tmp30 = triton_helpers.max2(tmp29, 1)[:, None]
    tmp31 = tmp26 - tmp30
    tmp32 = tl.exp(tmp31)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp32 / tmp36
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp37, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tj/ctj5eycg2odyg6423eehdipeo2ila5mtekoh247wu756pvldt3w5.py
# Source Nodes: [reshape_26], Original ATen: [aten.clone]
# reshape_26 => clone_16
triton_poi_fused_clone_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_94', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = (xindex // 32768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (65536 + x0 + (98304*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/cioildynrshhl23slbvjd6b5qg7abpsxbpqs5lyxu7mocmvl2xd4.py
# Source Nodes: [x_226], Original ATen: [aten._native_batch_norm_legit_functional]
# x_226 => var_mean_29
triton_red_fused__native_batch_norm_legit_functional_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
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
        tmp0 = tl.load(in_ptr0 + ((128*(((8*((r2 // 8) % 8)) + (r2 % 8)) % 64)) + (8192*((((8*((r2 // 8) % 8)) + (64*x0) + (32768*(r2 // 64)) + (65536*x1) + (r2 % 8)) // 8192) % 32)) + ((((8*((r2 // 8) % 8)) + (64*x0) + (r2 % 8)) // 64) % 128)), rmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nm4h2ktckoallwaqkl6h4za6dkiukh2qi52evcf6ag2p4jcwws.py
# Source Nodes: [x_226, x_229], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_226 => add_159, add_162, mul_206, mul_212, rsqrt_29, sub_32, var_mean_29
# x_229 => relu_25
triton_poi_fused__native_batch_norm_legit_functional_relu_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 64
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*x1) + (8192*((x1 + (64*x0)) // 8192)) + (32768*x2) + (x0 % 128)), None)
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
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23k4sbg4x4mjkk36x4o2pyjwwinvejf27mfe2ira45pucbllxn7.py
# Source Nodes: [x_231, x_237, x_238], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# x_231 => add_164, add_167, mul_213, mul_219, rsqrt_30, sub_33, var_mean_30
# x_237 => add_168
# x_238 => relu_26
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x2), tmp16, None)
    tl.store(out_ptr1 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hf/chfzikptu2wo2ulggxzypjq4ggz37rn6bqdq7fw5ld2y4gvcslae.py
# Source Nodes: [x_241, x_243], Original ATen: [aten.mean, aten.view]
# x_241 => mean
# x_243 => view_72
triton_per_fused_mean_view_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_98', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (131072*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckv62ou5ldlhemwyoghzfbcddsl3dpy4xbubixl2wxstecqzcdyq.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_99', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195 = args
    args.clear()
    assert_size_stride(primals_1, (24, ), (1, ))
    assert_size_stride(primals_2, (24, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (1024, ), (1, ))
    assert_size_stride(primals_40, (1024, ), (1, ))
    assert_size_stride(primals_41, (1024, ), (1, ))
    assert_size_stride(primals_42, (1024, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (31, 64), (64, 1))
    assert_size_stride(primals_46, (31, 64), (64, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_50, (1024, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (31, 128), (128, 1))
    assert_size_stride(primals_54, (31, 128), (128, 1))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (2048, ), (1, ))
    assert_size_stride(primals_58, (2048, ), (1, ))
    assert_size_stride(primals_59, (2048, ), (1, ))
    assert_size_stride(primals_60, (2048, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (15, 128), (128, 1))
    assert_size_stride(primals_64, (15, 128), (128, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (2048, ), (1, ))
    assert_size_stride(primals_68, (2048, ), (1, ))
    assert_size_stride(primals_69, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_70, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_71, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_72, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_74, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_75, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_76, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_77, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_78, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_79, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_81, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_82, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_83, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_84, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_85, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_86, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_87, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_88, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_89, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_90, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_91, (768, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_92, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_93, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_94, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_95, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_96, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_97, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_98, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_99, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_100, (1000, 2048), (2048, 1))
    assert_size_stride(primals_101, (1000, ), (1, ))
    assert_size_stride(primals_102, (), ())
    assert_size_stride(primals_103, (24, ), (1, ))
    assert_size_stride(primals_104, (24, ), (1, ))
    assert_size_stride(primals_105, (), ())
    assert_size_stride(primals_106, (32, ), (1, ))
    assert_size_stride(primals_107, (32, ), (1, ))
    assert_size_stride(primals_108, (), ())
    assert_size_stride(primals_109, (64, ), (1, ))
    assert_size_stride(primals_110, (64, ), (1, ))
    assert_size_stride(primals_111, (), ())
    assert_size_stride(primals_112, (64, ), (1, ))
    assert_size_stride(primals_113, (64, ), (1, ))
    assert_size_stride(primals_114, (), ())
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_116, (64, ), (1, ))
    assert_size_stride(primals_117, (), ())
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (), ())
    assert_size_stride(primals_121, (256, ), (1, ))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_123, (), ())
    assert_size_stride(primals_124, (64, ), (1, ))
    assert_size_stride(primals_125, (64, ), (1, ))
    assert_size_stride(primals_126, (), ())
    assert_size_stride(primals_127, (64, ), (1, ))
    assert_size_stride(primals_128, (64, ), (1, ))
    assert_size_stride(primals_129, (), ())
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (), ())
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (), ())
    assert_size_stride(primals_136, (128, ), (1, ))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_138, (), ())
    assert_size_stride(primals_139, (512, ), (1, ))
    assert_size_stride(primals_140, (512, ), (1, ))
    assert_size_stride(primals_141, (), ())
    assert_size_stride(primals_142, (512, ), (1, ))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (), ())
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (128, ), (1, ))
    assert_size_stride(primals_147, (), ())
    assert_size_stride(primals_148, (128, ), (1, ))
    assert_size_stride(primals_149, (128, ), (1, ))
    assert_size_stride(primals_150, (), ())
    assert_size_stride(primals_151, (512, ), (1, ))
    assert_size_stride(primals_152, (512, ), (1, ))
    assert_size_stride(primals_153, (), ())
    assert_size_stride(primals_154, (256, ), (1, ))
    assert_size_stride(primals_155, (256, ), (1, ))
    assert_size_stride(primals_156, (), ())
    assert_size_stride(primals_157, (256, ), (1, ))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_159, (), ())
    assert_size_stride(primals_160, (1024, ), (1, ))
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_162, (), ())
    assert_size_stride(primals_163, (1024, ), (1, ))
    assert_size_stride(primals_164, (1024, ), (1, ))
    assert_size_stride(primals_165, (), ())
    assert_size_stride(primals_166, (256, ), (1, ))
    assert_size_stride(primals_167, (256, ), (1, ))
    assert_size_stride(primals_168, (), ())
    assert_size_stride(primals_169, (256, ), (1, ))
    assert_size_stride(primals_170, (256, ), (1, ))
    assert_size_stride(primals_171, (), ())
    assert_size_stride(primals_172, (1024, ), (1, ))
    assert_size_stride(primals_173, (1024, ), (1, ))
    assert_size_stride(primals_174, (), ())
    assert_size_stride(primals_175, (512, ), (1, ))
    assert_size_stride(primals_176, (512, ), (1, ))
    assert_size_stride(primals_177, (), ())
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (), ())
    assert_size_stride(primals_181, (2048, ), (1, ))
    assert_size_stride(primals_182, (2048, ), (1, ))
    assert_size_stride(primals_183, (), ())
    assert_size_stride(primals_184, (2048, ), (1, ))
    assert_size_stride(primals_185, (2048, ), (1, ))
    assert_size_stride(primals_186, (), ())
    assert_size_stride(primals_187, (512, ), (1, ))
    assert_size_stride(primals_188, (512, ), (1, ))
    assert_size_stride(primals_189, (), ())
    assert_size_stride(primals_190, (512, ), (1, ))
    assert_size_stride(primals_191, (512, ), (1, ))
    assert_size_stride(primals_192, (), ())
    assert_size_stride(primals_193, (2048, ), (1, ))
    assert_size_stride(primals_194, (2048, ), (1, ))
    assert_size_stride(primals_195, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_69, buf0, 72, 9, grid=grid(72, 9), stream=stream0)
        del primals_69
        buf1 = empty_strided((32, 24, 3, 3), (216, 1, 72, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_70, buf1, 768, 9, grid=grid(768, 9), stream=stream0)
        del primals_70
        buf2 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_71, buf2, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_71
        buf3 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_73, buf3, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_73
        buf4 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_77, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_77
        buf5 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_80, buf5, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_80
        buf6 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_84, buf6, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_84
        buf7 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_87, buf7, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_87
        buf8 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_195, buf8, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del primals_195
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 24, 128, 128), (393216, 16384, 128, 1))
        buf10 = empty_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf9, buf10, 192, 16384, grid=grid(192, 16384), stream=stream0)
        buf11 = empty_strided((1, 24, 1, 1, 1024), (24576, 1, 24576, 24576, 24), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1, 24, 1, 1, 1024), (24576, 1, 24576, 24576, 24), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((1, 24, 1, 1, 1024), (24576, 1, 24576, 24576, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf10, buf11, buf12, buf13, 24576, 128, grid=grid(24576), stream=stream0)
        buf14 = empty_strided((1, 24, 1, 1, 8), (192, 1, 192, 192, 24), device='cuda', dtype=torch.float32)
        buf15 = empty_strided((1, 24, 1, 1, 8), (192, 1, 192, 192, 24), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((1, 24, 1, 1, 8), (192, 1, 192, 192, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf11, buf12, buf13, buf14, buf15, buf16, 192, 128, grid=grid(192), stream=stream0)
        del buf11
        del buf12
        del buf13
        buf17 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf18 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf20 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf14, buf15, buf16, primals_103, primals_104, buf17, buf18, buf20, primals_103, primals_104, 24, 8, grid=grid(24), stream=stream0)
        del buf14
        del buf15
        del buf16
        del primals_103
        del primals_104
        buf21 = reinterpret_tensor(buf9, (8, 24, 128, 128), (393216, 1, 3072, 24), 0); del buf9  # reuse
        # Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_11.run(buf10, buf17, buf18, primals_1, primals_2, buf21, 3145728, grid=grid(3145728), stream=stream0)
        del buf18
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 32, 128, 128), (524288, 16384, 128, 1))
        buf23 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf22, buf23, 256, 16384, grid=grid(256, 16384), stream=stream0)
        buf24 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        buf26 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf23, buf24, buf25, buf26, 32768, 128, grid=grid(32768), stream=stream0)
        buf27 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        buf28 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        buf29 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf24, buf25, buf26, buf27, buf28, buf29, 256, 128, grid=grid(256), stream=stream0)
        buf30 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf31 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf33 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf27, buf28, buf29, primals_106, primals_107, buf30, buf31, buf33, primals_106, primals_107, 32, 8, grid=grid(32), stream=stream0)
        del primals_106
        del primals_107
        buf34 = reinterpret_tensor(buf22, (8, 32, 128, 128), (524288, 1, 4096, 32), 0); del buf22  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf23, buf30, buf31, primals_3, primals_4, buf34, 4194304, grid=grid(4194304), stream=stream0)
        del buf31
        del primals_4
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        buf36 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf35, buf36, 512, 16384, grid=grid(512, 16384), stream=stream0)
        buf37 = empty_strided((1, 64, 1, 1, 1024), (65536, 1, 65536, 65536, 64), device='cuda', dtype=torch.float32)
        buf38 = empty_strided((1, 64, 1, 1, 1024), (65536, 1, 65536, 65536, 64), device='cuda', dtype=torch.float32)
        buf39 = empty_strided((1, 64, 1, 1, 1024), (65536, 1, 65536, 65536, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf36, buf37, buf38, buf39, 65536, 128, grid=grid(65536), stream=stream0)
        buf40 = empty_strided((1, 64, 1, 1, 8), (512, 1, 512, 512, 64), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((1, 64, 1, 1, 8), (512, 1, 512, 512, 64), device='cuda', dtype=torch.float32)
        buf42 = empty_strided((1, 64, 1, 1, 8), (512, 1, 512, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf37, buf38, buf39, buf40, buf41, buf42, 512, 128, grid=grid(512), stream=stream0)
        buf43 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf44 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf46 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf40, buf41, buf42, primals_109, primals_110, buf43, buf44, buf46, primals_109, primals_110, 64, 8, grid=grid(64), stream=stream0)
        del primals_109
        del primals_110
        buf47 = reinterpret_tensor(buf35, (8, 64, 128, 128), (1048576, 1, 8192, 64), 0); del buf35  # reuse
        # Source Nodes: [x_11, x_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_21.run(buf36, buf43, buf44, primals_5, primals_6, buf47, 8388608, grid=grid(8388608), stream=stream0)
        del primals_6
        buf48 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [shortcut], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_22.run(buf47, buf48, buf49, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf48, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf51 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf50, buf51, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf52 = empty_strided((1, 64, 1, 1, 256), (16384, 1, 16384, 16384, 64), device='cuda', dtype=torch.float32)
        buf53 = empty_strided((1, 64, 1, 1, 256), (16384, 1, 16384, 16384, 64), device='cuda', dtype=torch.float32)
        buf54 = empty_strided((1, 64, 1, 1, 256), (16384, 1, 16384, 16384, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf51, buf52, buf53, buf54, 16384, 128, grid=grid(16384), stream=stream0)
        buf55 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf56 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf57 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf52, buf53, buf54, buf55, buf56, buf57, 128, 128, grid=grid(128), stream=stream0)
        buf58 = buf44; del buf44  # reuse
        buf59 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf61 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf55, buf56, buf57, primals_112, primals_113, buf58, buf59, buf61, primals_112, primals_113, 64, 2, grid=grid(64), stream=stream0)
        del primals_112
        del primals_113
        buf62 = reinterpret_tensor(buf50, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf50  # reuse
        # Source Nodes: [x_17, x_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf51, buf58, buf59, primals_7, primals_8, buf62, 2097152, grid=grid(2097152), stream=stream0)
        del primals_8
        # Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf64 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf63, buf64, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf65 = buf54; del buf54  # reuse
        buf66 = buf53; del buf53  # reuse
        buf67 = buf52; del buf52  # reuse
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf64, buf65, buf66, buf67, 16384, 128, grid=grid(16384), stream=stream0)
        buf68 = buf57; del buf57  # reuse
        buf69 = buf56; del buf56  # reuse
        buf70 = buf55; del buf55  # reuse
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf65, buf66, buf67, buf68, buf69, buf70, 128, 128, grid=grid(128), stream=stream0)
        buf71 = buf59; del buf59  # reuse
        buf72 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf74 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf68, buf69, buf70, primals_115, primals_116, buf71, buf72, buf74, primals_115, primals_116, 64, 2, grid=grid(64), stream=stream0)
        del primals_115
        del primals_116
        buf75 = reinterpret_tensor(buf63, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf63  # reuse
        # Source Nodes: [x_23, x_27], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf64, buf71, buf72, primals_9, primals_10, buf75, 2097152, grid=grid(2097152), stream=stream0)
        del primals_10
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf77 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf76, buf77, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        buf78 = reinterpret_tensor(buf39, (1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), 0); del buf39  # reuse
        buf79 = reinterpret_tensor(buf38, (1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), 0); del buf38  # reuse
        buf80 = reinterpret_tensor(buf37, (1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), 0); del buf37  # reuse
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf77, buf78, buf79, buf80, 65536, 128, grid=grid(65536), stream=stream0)
        buf81 = reinterpret_tensor(buf42, (1, 256, 1, 1, 2), (512, 1, 512, 512, 256), 0); del buf42  # reuse
        buf82 = reinterpret_tensor(buf41, (1, 256, 1, 1, 2), (512, 1, 512, 512, 256), 0); del buf41  # reuse
        buf83 = reinterpret_tensor(buf40, (1, 256, 1, 1, 2), (512, 1, 512, 512, 256), 0); del buf40  # reuse
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf78, buf79, buf80, buf81, buf82, buf83, 512, 128, grid=grid(512), stream=stream0)
        buf84 = reinterpret_tensor(buf29, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf29  # reuse
        buf85 = reinterpret_tensor(buf28, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf28  # reuse
        buf87 = reinterpret_tensor(buf27, (256, ), (1, ), 0); del buf27  # reuse
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf81, buf82, buf83, primals_118, primals_119, buf84, buf85, buf87, primals_118, primals_119, 256, 2, grid=grid(256), stream=stream0)
        del primals_118
        del primals_119
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf48, primals_75, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf89 = reinterpret_tensor(buf76, (8, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf76  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf88, buf89, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        buf90 = buf80; del buf80  # reuse
        buf91 = buf79; del buf79  # reuse
        buf92 = buf78; del buf78  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf89, buf90, buf91, buf92, 65536, 128, grid=grid(65536), stream=stream0)
        buf93 = buf83; del buf83  # reuse
        buf94 = buf82; del buf82  # reuse
        buf95 = buf81; del buf81  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf90, buf91, buf92, buf93, buf94, buf95, 512, 128, grid=grid(512), stream=stream0)
        buf96 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf97 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf99 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf93, buf94, buf95, primals_121, primals_122, buf96, buf97, buf99, primals_121, primals_122, 256, 2, grid=grid(256), stream=stream0)
        del primals_121
        del primals_122
        buf100 = reinterpret_tensor(buf88, (8, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf88  # reuse
        buf101 = buf100; del buf100  # reuse
        # Source Nodes: [shortcut_1, x_31, x_39, x_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_32.run(buf101, buf77, buf84, buf85, primals_11, primals_12, buf89, buf96, buf97, primals_13, primals_14, 8388608, grid=grid(8388608), stream=stream0)
        del primals_12
        del primals_14
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf103 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf102, buf103, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf104 = buf67; del buf67  # reuse
        buf105 = buf66; del buf66  # reuse
        buf106 = buf65; del buf65  # reuse
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf103, buf104, buf105, buf106, 16384, 128, grid=grid(16384), stream=stream0)
        buf107 = buf70; del buf70  # reuse
        buf108 = buf69; del buf69  # reuse
        buf109 = buf68; del buf68  # reuse
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf104, buf105, buf106, buf107, buf108, buf109, 128, 128, grid=grid(128), stream=stream0)
        buf110 = buf72; del buf72  # reuse
        buf111 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf113 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf107, buf108, buf109, primals_124, primals_125, buf110, buf111, buf113, primals_124, primals_125, 64, 2, grid=grid(64), stream=stream0)
        del primals_124
        del primals_125
        buf114 = reinterpret_tensor(buf102, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf102  # reuse
        # Source Nodes: [x_45, x_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf103, buf110, buf111, primals_15, primals_16, buf114, 2097152, grid=grid(2097152), stream=stream0)
        del primals_16
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf116 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf115, buf116, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf117 = buf106; del buf106  # reuse
        buf118 = buf105; del buf105  # reuse
        buf119 = buf104; del buf104  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf116, buf117, buf118, buf119, 16384, 128, grid=grid(16384), stream=stream0)
        buf120 = buf109; del buf109  # reuse
        buf121 = buf108; del buf108  # reuse
        buf122 = buf107; del buf107  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf117, buf118, buf119, buf120, buf121, buf122, 128, 128, grid=grid(128), stream=stream0)
        buf123 = buf111; del buf111  # reuse
        buf124 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf126 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf120, buf121, buf122, primals_127, primals_128, buf123, buf124, buf126, primals_127, primals_128, 64, 2, grid=grid(64), stream=stream0)
        del primals_127
        del primals_128
        buf127 = reinterpret_tensor(buf115, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf115  # reuse
        # Source Nodes: [x_51, x_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf116, buf123, buf124, primals_17, primals_18, buf127, 2097152, grid=grid(2097152), stream=stream0)
        del buf124
        del primals_18
        # Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf129 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_58], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf128, buf129, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        buf130 = buf92; del buf92  # reuse
        buf131 = buf91; del buf91  # reuse
        buf132 = buf90; del buf90  # reuse
        # Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf129, buf130, buf131, buf132, 65536, 128, grid=grid(65536), stream=stream0)
        buf133 = buf95; del buf95  # reuse
        buf134 = buf94; del buf94  # reuse
        buf135 = buf93; del buf93  # reuse
        # Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf130, buf131, buf132, buf133, buf134, buf135, 512, 128, grid=grid(512), stream=stream0)
        del buf130
        del buf131
        del buf132
        buf136 = buf97; del buf97  # reuse
        buf137 = buf85; del buf85  # reuse
        buf139 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf133, buf134, buf135, primals_130, primals_131, buf136, buf137, buf139, primals_130, primals_131, 256, 2, grid=grid(256), stream=stream0)
        del primals_130
        del primals_131
        buf140 = reinterpret_tensor(buf128, (8, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf128  # reuse
        # Source Nodes: [shortcut_2, x_59, x_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_33.run(buf129, buf136, buf137, primals_19, primals_20, buf101, buf140, 8388608, grid=grid(8388608), stream=stream0)
        del primals_20
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf142 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf141, buf142, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        buf143 = reinterpret_tensor(buf26, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf26  # reuse
        buf144 = reinterpret_tensor(buf25, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf25  # reuse
        buf145 = reinterpret_tensor(buf24, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf24  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf142, buf143, buf144, buf145, 32768, 128, grid=grid(32768), stream=stream0)
        buf146 = reinterpret_tensor(buf137, (1, 128, 1, 1, 2), (256, 1, 256, 256, 128), 0); del buf137  # reuse
        buf147 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        buf148 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf143, buf144, buf145, buf146, buf147, buf148, 256, 128, grid=grid(256), stream=stream0)
        buf149 = reinterpret_tensor(buf122, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf122  # reuse
        buf150 = reinterpret_tensor(buf121, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf121  # reuse
        buf152 = reinterpret_tensor(buf120, (128, ), (1, ), 0); del buf120  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf146, buf147, buf148, primals_133, primals_134, buf149, buf150, buf152, primals_133, primals_134, 128, 2, grid=grid(128), stream=stream0)
        del primals_133
        del primals_134
        buf153 = reinterpret_tensor(buf141, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf141  # reuse
        # Source Nodes: [x_68, x_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf142, buf149, buf150, primals_21, primals_22, buf153, 4194304, grid=grid(4194304), stream=stream0)
        del primals_22
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf155 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf154, buf155, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf156 = empty_strided((1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), device='cuda', dtype=torch.float32)
        buf157 = empty_strided((1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), device='cuda', dtype=torch.float32)
        buf158 = empty_strided((1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf155, buf156, buf157, buf158, 8192, 128, grid=grid(8192), stream=stream0)
        buf159 = buf150; del buf150  # reuse
        buf160 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf162 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf156, buf157, buf158, primals_136, primals_137, buf159, buf160, buf162, primals_136, primals_137, 128, 64, grid=grid(128), stream=stream0)
        del primals_136
        del primals_137
        buf163 = reinterpret_tensor(buf154, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf154  # reuse
        # Source Nodes: [x_74, x_78], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_42.run(buf155, buf159, buf160, primals_23, primals_24, buf163, 1048576, grid=grid(1048576), stream=stream0)
        del primals_24
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf165 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf164, buf165, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        buf166 = reinterpret_tensor(buf145, (1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), 0); del buf145  # reuse
        buf167 = reinterpret_tensor(buf144, (1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), 0); del buf144  # reuse
        buf168 = reinterpret_tensor(buf143, (1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), 0); del buf143  # reuse
        # Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf165, buf166, buf167, buf168, 32768, 128, grid=grid(32768), stream=stream0)
        buf169 = reinterpret_tensor(buf135, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf135  # reuse
        buf170 = reinterpret_tensor(buf134, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf134  # reuse
        buf172 = reinterpret_tensor(buf133, (512, ), (1, ), 0); del buf133  # reuse
        # Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf166, buf167, buf168, primals_139, primals_140, buf169, buf170, buf172, primals_139, primals_140, 512, 64, grid=grid(512), stream=stream0)
        del primals_139
        del primals_140
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf140, primals_82, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf174 = reinterpret_tensor(buf164, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf164  # reuse
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf173, buf174, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        buf175 = buf168; del buf168  # reuse
        buf176 = buf167; del buf167  # reuse
        buf177 = buf166; del buf166  # reuse
        # Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf174, buf175, buf176, buf177, 32768, 128, grid=grid(32768), stream=stream0)
        buf178 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf179 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf181 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf175, buf176, buf177, primals_142, primals_143, buf178, buf179, buf181, primals_142, primals_143, 512, 64, grid=grid(512), stream=stream0)
        del primals_142
        del primals_143
        buf182 = reinterpret_tensor(buf173, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf173  # reuse
        buf183 = buf182; del buf182  # reuse
        # Source Nodes: [shortcut_3, x_82, x_90, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_46.run(buf183, buf165, buf169, buf170, primals_25, primals_26, buf174, buf178, buf179, primals_27, primals_28, 4194304, grid=grid(4194304), stream=stream0)
        del primals_26
        del primals_28
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_83, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf185 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf184, buf185, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf186 = buf158; del buf158  # reuse
        buf187 = buf157; del buf157  # reuse
        buf188 = buf156; del buf156  # reuse
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf185, buf186, buf187, buf188, 8192, 128, grid=grid(8192), stream=stream0)
        buf189 = buf160; del buf160  # reuse
        buf190 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf192 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf186, buf187, buf188, primals_145, primals_146, buf189, buf190, buf192, primals_145, primals_146, 128, 64, grid=grid(128), stream=stream0)
        del primals_145
        del primals_146
        buf193 = reinterpret_tensor(buf184, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf184  # reuse
        # Source Nodes: [x_100, x_96], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_42.run(buf185, buf189, buf190, primals_29, primals_30, buf193, 1048576, grid=grid(1048576), stream=stream0)
        del primals_30
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf195 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf194, buf195, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf196 = buf188; del buf188  # reuse
        buf197 = buf187; del buf187  # reuse
        buf198 = buf186; del buf186  # reuse
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf195, buf196, buf197, buf198, 8192, 128, grid=grid(8192), stream=stream0)
        buf199 = buf190; del buf190  # reuse
        buf200 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf202 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf196, buf197, buf198, primals_148, primals_149, buf199, buf200, buf202, primals_148, primals_149, 128, 64, grid=grid(128), stream=stream0)
        del primals_148
        del primals_149
        buf203 = reinterpret_tensor(buf194, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf194  # reuse
        # Source Nodes: [x_102, x_106], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_42.run(buf195, buf199, buf200, primals_31, primals_32, buf203, 1048576, grid=grid(1048576), stream=stream0)
        del buf200
        del primals_32
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf205 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf204, buf205, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        buf206 = buf177; del buf177  # reuse
        buf207 = buf176; del buf176  # reuse
        buf208 = buf175; del buf175  # reuse
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf205, buf206, buf207, buf208, 32768, 128, grid=grid(32768), stream=stream0)
        buf209 = buf179; del buf179  # reuse
        buf210 = buf170; del buf170  # reuse
        buf212 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf206, buf207, buf208, primals_151, primals_152, buf209, buf210, buf212, primals_151, primals_152, 512, 64, grid=grid(512), stream=stream0)
        del buf206
        del buf207
        del buf208
        del primals_151
        del primals_152
        buf213 = reinterpret_tensor(buf204, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf204  # reuse
        # Source Nodes: [shortcut_4, x_110, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_47.run(buf205, buf209, buf210, primals_33, primals_34, buf183, buf213, 4194304, grid=grid(4194304), stream=stream0)
        del primals_34
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf215 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48.run(buf214, buf215, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf216 = reinterpret_tensor(buf119, (1, 256, 1, 1, 64), (16384, 1, 16384, 16384, 256), 0); del buf119  # reuse
        buf217 = reinterpret_tensor(buf118, (1, 256, 1, 1, 64), (16384, 1, 16384, 16384, 256), 0); del buf118  # reuse
        buf218 = reinterpret_tensor(buf117, (1, 256, 1, 1, 64), (16384, 1, 16384, 16384, 256), 0); del buf117  # reuse
        # Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_49.run(buf215, buf216, buf217, buf218, 16384, 128, grid=grid(16384), stream=stream0)
        buf219 = reinterpret_tensor(buf148, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf148  # reuse
        buf220 = reinterpret_tensor(buf147, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf147  # reuse
        buf222 = reinterpret_tensor(buf146, (256, ), (1, ), 0); del buf146  # reuse
        # Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_50.run(buf216, buf217, buf218, primals_154, primals_155, buf219, buf220, buf222, primals_154, primals_155, 256, 64, grid=grid(256), stream=stream0)
        del primals_154
        del primals_155
        buf223 = reinterpret_tensor(buf214, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf214  # reuse
        # Source Nodes: [x_119, x_123], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_51.run(buf215, buf219, buf220, primals_35, primals_36, buf223, 2097152, grid=grid(2097152), stream=stream0)
        del primals_36
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf225 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf224, buf225, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf226 = empty_strided((1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), device='cuda', dtype=torch.float32)
        buf227 = empty_strided((1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), device='cuda', dtype=torch.float32)
        buf228 = empty_strided((1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf225, buf226, buf227, buf228, 4096, 128, grid=grid(4096), stream=stream0)
        buf229 = buf220; del buf220  # reuse
        buf230 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf232 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf226, buf227, buf228, primals_157, primals_158, buf229, buf230, buf232, primals_157, primals_158, 256, 16, grid=grid(256), stream=stream0)
        del primals_157
        del primals_158
        buf233 = reinterpret_tensor(buf224, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf224  # reuse
        # Source Nodes: [x_125, x_129], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf225, buf229, buf230, primals_37, primals_38, buf233, 524288, grid=grid(524288), stream=stream0)
        del primals_38
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf235 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf234, buf235, 8192, 256, grid=grid(8192, 256), stream=stream0)
        buf236 = reinterpret_tensor(buf218, (1, 1024, 1, 1, 16), (16384, 1, 16384, 16384, 1024), 0); del buf218  # reuse
        buf237 = reinterpret_tensor(buf217, (1, 1024, 1, 1, 16), (16384, 1, 16384, 16384, 1024), 0); del buf217  # reuse
        buf238 = reinterpret_tensor(buf216, (1, 1024, 1, 1, 16), (16384, 1, 16384, 16384, 1024), 0); del buf216  # reuse
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf235, buf236, buf237, buf238, 16384, 128, grid=grid(16384), stream=stream0)
        buf239 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf240 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf242 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf236, buf237, buf238, primals_160, primals_161, buf239, buf240, buf242, primals_160, primals_161, 1024, 16, grid=grid(1024), stream=stream0)
        del primals_160
        del primals_161
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf213, primals_89, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf244 = reinterpret_tensor(buf234, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf234  # reuse
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf243, buf244, 8192, 256, grid=grid(8192, 256), stream=stream0)
        buf245 = buf238; del buf238  # reuse
        buf246 = buf237; del buf237  # reuse
        buf247 = buf236; del buf236  # reuse
        # Source Nodes: [x_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf244, buf245, buf246, buf247, 16384, 128, grid=grid(16384), stream=stream0)
        buf248 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf249 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf251 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf245, buf246, buf247, primals_163, primals_164, buf248, buf249, buf251, primals_163, primals_164, 1024, 16, grid=grid(1024), stream=stream0)
        del primals_163
        del primals_164
        buf252 = reinterpret_tensor(buf243, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf243  # reuse
        buf253 = buf252; del buf252  # reuse
        # Source Nodes: [shortcut_5, x_133, x_141, x_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_59.run(buf253, buf235, buf239, buf240, primals_39, primals_40, buf244, buf248, buf249, primals_41, primals_42, 2097152, grid=grid(2097152), stream=stream0)
        del primals_40
        del primals_42
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf255 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf254, buf255, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf256 = buf228; del buf228  # reuse
        buf257 = buf227; del buf227  # reuse
        buf258 = buf226; del buf226  # reuse
        # Source Nodes: [x_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf255, buf256, buf257, buf258, 4096, 128, grid=grid(4096), stream=stream0)
        buf259 = buf230; del buf230  # reuse
        buf260 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf262 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf256, buf257, buf258, primals_166, primals_167, buf259, buf260, buf262, primals_166, primals_167, 256, 16, grid=grid(256), stream=stream0)
        del primals_166
        del primals_167
        buf263 = reinterpret_tensor(buf254, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf254  # reuse
        # Source Nodes: [x_147, x_151], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf255, buf259, buf260, primals_43, primals_44, buf263, 524288, grid=grid(524288), stream=stream0)
        del primals_44
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 768, 16, 16), (196608, 256, 16, 1))
        buf265 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape], Original ATen: [aten.clone]
        triton_poi_fused_clone_60.run(buf264, buf265, 524288, grid=grid(524288), stream=stream0)
        buf266 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_61.run(buf264, buf266, 524288, grid=grid(524288), stream=stream0)
        buf267 = empty((32, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (32, 256, 64), (16384, 1, 256), 0), reinterpret_tensor(buf266, (32, 64, 256), (16384, 256, 1), 0), out=buf267)
        buf268 = empty((8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_62.run(buf265, buf268, 8192, 64, grid=grid(8192, 64), stream=stream0)
        buf269 = empty((8192, 31), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.mm]
        extern_kernels.mm(buf268, reinterpret_tensor(primals_45, (64, 31), (1, 64), 0), out=buf269)
        buf270 = empty((8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_63.run(buf265, buf270, 524288, grid=grid(524288), stream=stream0)
        buf271 = empty((8192, 31), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.mm]
        extern_kernels.mm(buf270, reinterpret_tensor(primals_46, (64, 31), (1, 64), 0), out=buf271)
        buf274 = empty((32, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1, mul], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_64.run(buf267, buf271, buf269, buf274, 8192, 256, grid=grid(8192), stream=stream0)
        buf275 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_65.run(buf264, buf275, 524288, grid=grid(524288), stream=stream0)
        del buf264
        buf276 = empty((32, 256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf274, reinterpret_tensor(buf275, (32, 256, 64), (16384, 1, 256), 0), out=buf276)
        buf277 = buf258; del buf258  # reuse
        buf278 = buf257; del buf257  # reuse
        buf279 = buf256; del buf256  # reuse
        # Source Nodes: [x_163], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf276, buf277, buf278, buf279, 4096, 128, grid=grid(4096), stream=stream0)
        buf280 = buf260; del buf260  # reuse
        buf281 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf283 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf277, buf278, buf279, primals_169, primals_170, buf280, buf281, buf283, primals_169, primals_170, 256, 16, grid=grid(256), stream=stream0)
        del buf277
        del buf278
        del buf279
        del primals_169
        del primals_170
        buf284 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163, x_166], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_67.run(buf276, buf280, buf281, primals_47, primals_48, buf284, 524288, grid=grid(524288), stream=stream0)
        del buf281
        del primals_48
        # Source Nodes: [x_167], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf286 = reinterpret_tensor(buf267, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf267  # reuse
        # Source Nodes: [x_167], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf285, buf286, 8192, 256, grid=grid(8192, 256), stream=stream0)
        buf287 = buf247; del buf247  # reuse
        buf288 = buf246; del buf246  # reuse
        buf289 = buf245; del buf245  # reuse
        # Source Nodes: [x_168], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf286, buf287, buf288, buf289, 16384, 128, grid=grid(16384), stream=stream0)
        buf290 = buf249; del buf249  # reuse
        buf291 = buf240; del buf240  # reuse
        buf293 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf287, buf288, buf289, primals_172, primals_173, buf290, buf291, buf293, primals_172, primals_173, 1024, 16, grid=grid(1024), stream=stream0)
        del buf287
        del buf288
        del primals_172
        del primals_173
        buf294 = reinterpret_tensor(buf285, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf285  # reuse
        # Source Nodes: [shortcut_6, x_168, x_174], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_68.run(buf286, buf290, buf291, primals_49, primals_50, buf253, buf294, 2097152, grid=grid(2097152), stream=stream0)
        del buf291
        del primals_50
        # Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, primals_93, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf296 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf295, buf296, 4096, 256, grid=grid(4096, 256), stream=stream0)
        buf297 = reinterpret_tensor(buf198, (1, 512, 1, 1, 16), (8192, 1, 8192, 8192, 512), 0); del buf198  # reuse
        buf298 = reinterpret_tensor(buf197, (1, 512, 1, 1, 16), (8192, 1, 8192, 8192, 512), 0); del buf197  # reuse
        buf299 = reinterpret_tensor(buf196, (1, 512, 1, 1, 16), (8192, 1, 8192, 8192, 512), 0); del buf196  # reuse
        # Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_70.run(buf296, buf297, buf298, buf299, 8192, 128, grid=grid(8192), stream=stream0)
        buf300 = buf210; del buf210  # reuse
        buf301 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf303 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_71.run(buf297, buf298, buf299, primals_175, primals_176, buf300, buf301, buf303, primals_175, primals_176, 512, 16, grid=grid(512), stream=stream0)
        del primals_175
        del primals_176
        buf304 = reinterpret_tensor(buf295, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf295  # reuse
        # Source Nodes: [x_176, x_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_72.run(buf296, buf300, buf301, primals_51, primals_52, buf304, 1048576, grid=grid(1048576), stream=stream0)
        del primals_52
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (8, 1536, 16, 16), (393216, 256, 16, 1))
        buf306 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_73.run(buf305, buf306, 1048576, grid=grid(1048576), stream=stream0)
        buf307 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_74.run(buf305, buf307, 1048576, grid=grid(1048576), stream=stream0)
        buf308 = empty((32, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf306, (32, 256, 128), (32768, 1, 256), 0), reinterpret_tensor(buf307, (32, 128, 256), (32768, 256, 1), 0), out=buf308)
        buf309 = empty((8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_75.run(buf306, buf309, 8192, 128, grid=grid(8192, 128), stream=stream0)
        buf310 = buf271; del buf271  # reuse
        # Source Nodes: [x_183], Original ATen: [aten.mm]
        extern_kernels.mm(buf309, reinterpret_tensor(primals_53, (128, 31), (1, 128), 0), out=buf310)
        buf311 = empty((8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_76.run(buf306, buf311, 1048576, grid=grid(1048576), stream=stream0)
        buf312 = buf269; del buf269  # reuse
        # Source Nodes: [x_187], Original ATen: [aten.mm]
        extern_kernels.mm(buf311, reinterpret_tensor(primals_54, (128, 31), (1, 128), 0), out=buf312)
        buf315 = empty((32, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_2, attn_3, mul_1], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_77.run(buf308, buf312, buf310, buf315, 8192, 256, grid=grid(8192), stream=stream0)
        del buf308
        del buf310
        del buf312
        buf316 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_78.run(buf305, buf316, 1048576, grid=grid(1048576), stream=stream0)
        del buf305
        buf317 = empty((32, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf315, reinterpret_tensor(buf316, (32, 256, 128), (32768, 1, 256), 0), out=buf317)
        buf318 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_2], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_79.run(buf317, buf318, 1048576, grid=grid(1048576), stream=stream0)
        buf319 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_80.run(buf318, buf319, 262144, grid=grid(262144), stream=stream0)
        buf320 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        buf321 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        buf322 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf319, buf320, buf321, buf322, 2048, 128, grid=grid(2048), stream=stream0)
        buf323 = buf301; del buf301  # reuse
        buf324 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf326 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf320, buf321, buf322, primals_178, primals_179, buf323, buf324, buf326, primals_178, primals_179, 512, 4, grid=grid(512), stream=stream0)
        del primals_178
        del primals_179
        buf327 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_192, x_195], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_83.run(buf319, buf323, buf324, primals_55, primals_56, buf327, 262144, grid=grid(262144), stream=stream0)
        del primals_56
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, primals_95, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (8, 2048, 8, 8), (131072, 64, 8, 1))
        buf329 = reinterpret_tensor(buf317, (8, 2048, 8, 8), (131072, 1, 16384, 2048), 0); del buf317  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_84.run(buf328, buf329, 16384, 64, grid=grid(16384, 64), stream=stream0)
        buf330 = reinterpret_tensor(buf299, (1, 2048, 1, 1, 4), (8192, 1, 8192, 8192, 2048), 0); del buf299  # reuse
        buf331 = reinterpret_tensor(buf298, (1, 2048, 1, 1, 4), (8192, 1, 8192, 8192, 2048), 0); del buf298  # reuse
        buf332 = reinterpret_tensor(buf297, (1, 2048, 1, 1, 4), (8192, 1, 8192, 8192, 2048), 0); del buf297  # reuse
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_85.run(buf329, buf330, buf331, buf332, 8192, 128, grid=grid(8192), stream=stream0)
        buf333 = reinterpret_tensor(buf322, (1, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf322  # reuse
        buf334 = reinterpret_tensor(buf321, (1, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf321  # reuse
        buf336 = reinterpret_tensor(buf320, (2048, ), (1, ), 0); del buf320  # reuse
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_86.run(buf330, buf331, buf332, primals_181, primals_182, buf333, buf334, buf336, primals_181, primals_182, 2048, 4, grid=grid(2048), stream=stream0)
        del primals_181
        del primals_182
        # Source Nodes: [x_203], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf294, primals_96, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (8, 2048, 8, 8), (131072, 64, 8, 1))
        buf338 = reinterpret_tensor(buf328, (8, 2048, 8, 8), (131072, 1, 16384, 2048), 0); del buf328  # reuse
        # Source Nodes: [x_203], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_84.run(buf337, buf338, 16384, 64, grid=grid(16384, 64), stream=stream0)
        buf339 = buf332; del buf332  # reuse
        buf340 = buf331; del buf331  # reuse
        buf341 = buf330; del buf330  # reuse
        # Source Nodes: [x_204], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_85.run(buf338, buf339, buf340, buf341, 8192, 128, grid=grid(8192), stream=stream0)
        buf342 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf343 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf345 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_86.run(buf339, buf340, buf341, primals_184, primals_185, buf342, buf343, buf345, primals_184, primals_185, 2048, 4, grid=grid(2048), stream=stream0)
        del primals_184
        del primals_185
        buf346 = reinterpret_tensor(buf337, (8, 2048, 8, 8), (131072, 1, 16384, 2048), 0); del buf337  # reuse
        buf347 = buf346; del buf346  # reuse
        # Source Nodes: [shortcut_7, x_197, x_204, x_208], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_87.run(buf347, buf329, buf333, buf334, primals_57, primals_58, buf338, buf342, buf343, primals_59, primals_60, 1048576, grid=grid(1048576), stream=stream0)
        del primals_58
        del primals_60
        # Source Nodes: [x_209], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf347, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf349 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_209], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf348, buf349, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf350 = reinterpret_tensor(buf343, (1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), 0); del buf343  # reuse
        buf351 = reinterpret_tensor(buf334, (1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), 0); del buf334  # reuse
        buf352 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf349, buf350, buf351, buf352, 2048, 128, grid=grid(2048), stream=stream0)
        buf353 = buf324; del buf324  # reuse
        buf354 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf356 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf350, buf351, buf352, primals_187, primals_188, buf353, buf354, buf356, primals_187, primals_188, 512, 4, grid=grid(512), stream=stream0)
        del primals_187
        del primals_188
        buf357 = reinterpret_tensor(buf348, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf348  # reuse
        # Source Nodes: [x_210, x_214], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_83.run(buf349, buf353, buf354, primals_61, primals_62, buf357, 262144, grid=grid(262144), stream=stream0)
        del primals_62
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (8, 1536, 8, 8), (98304, 64, 8, 1))
        buf359 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_89.run(buf358, buf359, 262144, grid=grid(262144), stream=stream0)
        buf360 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_90.run(buf358, buf360, 262144, grid=grid(262144), stream=stream0)
        buf361 = empty((32, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf359, (32, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf360, (32, 128, 64), (8192, 64, 1), 0), out=buf361)
        buf362 = empty((2048, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_91.run(buf359, buf362, 2048, 128, grid=grid(2048, 128), stream=stream0)
        buf363 = empty((2048, 15), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217], Original ATen: [aten.mm]
        extern_kernels.mm(buf362, reinterpret_tensor(primals_63, (128, 15), (1, 128), 0), out=buf363)
        buf364 = empty((2048, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_92.run(buf359, buf364, 262144, grid=grid(262144), stream=stream0)
        buf365 = empty((2048, 15), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten.mm]
        extern_kernels.mm(buf364, reinterpret_tensor(primals_64, (128, 15), (1, 128), 0), out=buf365)
        buf368 = empty((32, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_4, attn_5, mul_2], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_93.run(buf361, buf365, buf363, buf368, 2048, 64, grid=grid(2048), stream=stream0)
        del buf361
        del buf363
        del buf365
        buf369 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_94.run(buf358, buf369, 262144, grid=grid(262144), stream=stream0)
        del buf358
        buf370 = empty((32, 64, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf368, reinterpret_tensor(buf369, (32, 64, 128), (8192, 1, 64), 0), out=buf370)
        buf371 = buf352; del buf352  # reuse
        buf372 = buf351; del buf351  # reuse
        buf373 = buf350; del buf350  # reuse
        # Source Nodes: [x_226], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_95.run(buf370, buf371, buf372, buf373, 2048, 128, grid=grid(2048), stream=stream0)
        buf374 = buf354; del buf354  # reuse
        buf375 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf377 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf371, buf372, buf373, primals_190, primals_191, buf374, buf375, buf377, primals_190, primals_191, 512, 4, grid=grid(512), stream=stream0)
        del primals_190
        del primals_191
        buf378 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226, x_229], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_96.run(buf370, buf374, buf375, primals_65, primals_66, buf378, 262144, grid=grid(262144), stream=stream0)
        del buf375
        del primals_66
        # Source Nodes: [x_230], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_99, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (8, 2048, 8, 8), (131072, 64, 8, 1))
        buf380 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_230], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_84.run(buf379, buf380, 16384, 64, grid=grid(16384, 64), stream=stream0)
        buf381 = buf341; del buf341  # reuse
        buf382 = buf340; del buf340  # reuse
        buf383 = buf339; del buf339  # reuse
        # Source Nodes: [x_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_85.run(buf380, buf381, buf382, buf383, 8192, 128, grid=grid(8192), stream=stream0)
        buf384 = reinterpret_tensor(buf373, (1, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf373  # reuse
        buf385 = reinterpret_tensor(buf372, (1, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf372  # reuse
        buf387 = reinterpret_tensor(buf371, (2048, ), (1, ), 0); del buf371  # reuse
        # Source Nodes: [x_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_86.run(buf381, buf382, buf383, primals_193, primals_194, buf384, buf385, buf387, primals_193, primals_194, 2048, 4, grid=grid(2048), stream=stream0)
        del buf381
        del buf382
        del buf383
        del primals_193
        del primals_194
        buf388 = reinterpret_tensor(buf379, (8, 2048, 8, 8), (131072, 1, 16384, 2048), 0); del buf379  # reuse
        buf392 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_231, x_237, x_238], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_97.run(buf380, buf384, buf385, primals_67, primals_68, buf347, buf388, buf392, 1048576, grid=grid(1048576), stream=stream0)
        del buf385
        del primals_68
        buf389 = reinterpret_tensor(buf289, (8, 2048, 1, 1), (2048, 1, 16384, 16384), 0); del buf289  # reuse
        buf390 = reinterpret_tensor(buf389, (8, 2048), (2048, 1), 0); del buf389  # reuse
        # Source Nodes: [x_241, x_243], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_98.run(buf390, buf388, 16384, 64, grid=grid(16384), stream=stream0)
        del buf388
        buf391 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_101, buf390, reinterpret_tensor(primals_100, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf391)
        del primals_101
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_102, primals_102, 1, grid=grid(1), stream=stream0)
        del primals_102
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_105, primals_105, 1, grid=grid(1), stream=stream0)
        del primals_105
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_108, primals_108, 1, grid=grid(1), stream=stream0)
        del primals_108
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_111, primals_111, 1, grid=grid(1), stream=stream0)
        del primals_111
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_114, primals_114, 1, grid=grid(1), stream=stream0)
        del primals_114
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_117, primals_117, 1, grid=grid(1), stream=stream0)
        del primals_117
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_120, primals_120, 1, grid=grid(1), stream=stream0)
        del primals_120
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_123, primals_123, 1, grid=grid(1), stream=stream0)
        del primals_123
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_126, primals_126, 1, grid=grid(1), stream=stream0)
        del primals_126
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_129, primals_129, 1, grid=grid(1), stream=stream0)
        del primals_129
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_132, primals_132, 1, grid=grid(1), stream=stream0)
        del primals_132
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_135, primals_135, 1, grid=grid(1), stream=stream0)
        del primals_135
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_138, primals_138, 1, grid=grid(1), stream=stream0)
        del primals_138
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_141, primals_141, 1, grid=grid(1), stream=stream0)
        del primals_141
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_144, primals_144, 1, grid=grid(1), stream=stream0)
        del primals_144
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_147, primals_147, 1, grid=grid(1), stream=stream0)
        del primals_147
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_150, primals_150, 1, grid=grid(1), stream=stream0)
        del primals_150
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_153, primals_153, 1, grid=grid(1), stream=stream0)
        del primals_153
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_156, primals_156, 1, grid=grid(1), stream=stream0)
        del primals_156
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_159, primals_159, 1, grid=grid(1), stream=stream0)
        del primals_159
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_162, primals_162, 1, grid=grid(1), stream=stream0)
        del primals_162
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_165, primals_165, 1, grid=grid(1), stream=stream0)
        del primals_165
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_168, primals_168, 1, grid=grid(1), stream=stream0)
        del primals_168
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_171, primals_171, 1, grid=grid(1), stream=stream0)
        del primals_171
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_174, primals_174, 1, grid=grid(1), stream=stream0)
        del primals_174
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_177, primals_177, 1, grid=grid(1), stream=stream0)
        del primals_177
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_180, primals_180, 1, grid=grid(1), stream=stream0)
        del primals_180
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_183, primals_183, 1, grid=grid(1), stream=stream0)
        del primals_183
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_186, primals_186, 1, grid=grid(1), stream=stream0)
        del primals_186
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_189, primals_189, 1, grid=grid(1), stream=stream0)
        del primals_189
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_99.run(primals_192, primals_192, 1, grid=grid(1), stream=stream0)
        del primals_192
        return (buf391, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, buf0, buf1, buf2, primals_72, buf3, primals_74, primals_75, primals_76, buf4, primals_78, primals_79, buf5, primals_81, primals_82, primals_83, buf6, primals_85, primals_86, buf7, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, buf8, buf10, buf20, buf21, buf23, buf33, buf34, buf36, buf46, buf47, buf48, buf49, buf51, buf61, buf62, buf64, buf74, buf75, buf77, buf87, buf89, buf99, buf101, buf103, buf113, buf114, buf116, buf126, buf127, buf129, buf139, buf140, buf142, buf152, buf153, buf155, buf162, buf163, buf165, buf172, buf174, buf181, buf183, buf185, buf192, buf193, buf195, buf202, buf203, buf205, buf212, buf213, buf215, buf222, buf223, buf225, buf232, buf233, buf235, buf242, buf244, buf251, buf253, buf255, buf262, buf263, buf268, buf270, buf276, buf283, buf284, buf286, buf293, buf294, buf296, buf303, buf304, buf309, buf311, buf318, buf319, buf326, buf327, buf329, buf336, buf338, buf345, buf347, buf349, buf356, buf357, buf362, buf364, buf370, buf377, buf378, buf380, buf387, buf390, reinterpret_tensor(primals_100, (1000, 2048), (2048, 1), 0), buf392, reinterpret_tensor(buf384, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf374, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf368, (32, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf369, (32, 128, 64), (8192, 64, 1), 0), buf368, reinterpret_tensor(primals_64, (15, 128), (128, 1), 0), reinterpret_tensor(primals_63, (15, 128), (128, 1), 0), reinterpret_tensor(buf359, (32, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf360, (32, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf353, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf342, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf333, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf323, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf315, (32, 256, 256), (65536, 1, 256), 0), reinterpret_tensor(buf316, (32, 128, 256), (32768, 256, 1), 0), buf315, reinterpret_tensor(primals_54, (31, 128), (128, 1), 0), reinterpret_tensor(primals_53, (31, 128), (128, 1), 0), reinterpret_tensor(buf306, (32, 128, 256), (32768, 256, 1), 0), reinterpret_tensor(buf307, (32, 256, 128), (32768, 1, 256), 0), reinterpret_tensor(buf300, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf290, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf280, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf274, (32, 256, 256), (65536, 1, 256), 0), reinterpret_tensor(buf275, (32, 64, 256), (16384, 256, 1), 0), buf274, reinterpret_tensor(primals_46, (31, 64), (64, 1), 0), reinterpret_tensor(primals_45, (31, 64), (64, 1), 0), reinterpret_tensor(buf265, (32, 64, 256), (16384, 256, 1), 0), reinterpret_tensor(buf266, (32, 256, 64), (16384, 1, 256), 0), reinterpret_tensor(buf259, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf248, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf239, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf229, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf219, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf209, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf199, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf189, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf178, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf169, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf159, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf149, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf136, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf123, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf110, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf96, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf84, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf71, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf58, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf43, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf30, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf17, (1, 24, 1, 1), (24, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((31, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((31, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((31, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((31, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((15, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((15, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_103 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_106 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_109 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_112 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_115 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_124 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_127 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_139 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_151 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_154 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_157 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_160 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_163 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_166 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_169 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_172 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_175 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_181 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_184 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_187 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_193 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('botnet26t_256', benchmark_compiled_module)
