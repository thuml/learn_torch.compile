
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


# kernel path: /tmp/torchinductor_youkaichao/xe/cxe2e7tfscjn4eprl2y4jezohfff7o456zv5sacr3v7teiresozk.py
# Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_4 => mul_7, sigmoid
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
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
# x_6 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_13, mul_9, rsqrt_1, squeeze_4, var_mean_1
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


# kernel path: /tmp/torchinductor_youkaichao/n2/cn2tdfhegjbsjln7e64kaftiuv2mdriruqrzlg6vjoj6ixc43igg.py
# Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_6 => add_6, add_9, mul_14, mul_8, rsqrt_1, sub_1, var_mean_1
# x_9 => mul_15, sigmoid_1
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpnjt2vwsrkowjhb2izqsmg7gaf27dwtyl55mmpxmhppytks53ay.py
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
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxjeqwujrawop72pjpsqjnrpvlrnpy4zzdosswgcyyaiztitshv.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/e7/ce7i2lrcppcptc2whvr6d4sst3u42px6rwgny4oynwwfn5redla5.py
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
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/up/cupdwmf7qn5hnyhcyxstaeaex2qvu5k2e7zhiuokdghqsxsbxgp4.py
# Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
# x_11 => add_11, add_12, add_13, mul_17, mul_18, mul_19, mul_20, mul_21, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/6v/c6vfmf26ijfzexx6mb44vbd4wxy6z27sxmecxtkrxfarwdwmldoc.py
# Source Nodes: [shortcut, x_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut => mul_23, sigmoid_2
# x_11 => add_11, add_14, mul_16, mul_22, rsqrt_2, sub_2, var_mean_2
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdliu6hiw7pctzcbl7xlwzntfamkcshjtthloyedf4eylgfkhwm.py
# Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
# x_23 => add_21, add_24, mul_32, mul_38, rsqrt_4, sub_4, var_mean_4
triton_poi_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vg/cvggfc4rypctokt7mg4kdti64cmo6hjx4sg4w365mrehsqz2cosa.py
# Source Nodes: [x_27, x_se], Original ATen: [aten.mean, aten.silu]
# x_27 => mul_39, sigmoid_4
# x_se => mean
triton_red_fused_mean_silu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/cckbr6i7eaf2ae4cyqa5gmujznmq7snxkdekzjj2agcbddumedxv.py
# Source Nodes: [x_27, x_se], Original ATen: [aten.mean, aten.silu]
# x_27 => mul_39, sigmoid_4
# x_se => mean
triton_per_fused_mean_silu_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (2048*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 4096.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbcjbfghxw5jzi25yrhumeh65ecask37j52e4tko6dplbejx4mq.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
# x_se_1 => convolution_5
# x_se_2 => relu
triton_poi_fused_convolution_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7l5u3hlyvxygykcgvtllqi5h2ajdpilasajcz6lr7uomiiyjjp5.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution]
# x_se_3 => convolution_6
triton_poi_fused_convolution_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4f/c4fe4jzaf564eilvrxqexmsvnotooe55arp3j5seejee35q45yva.py
# Source Nodes: [sigmoid, x_27, x_29], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# sigmoid => sigmoid_5
# x_27 => mul_39, sigmoid_4
# x_29 => mul_40
triton_poi_fused_mul_sigmoid_silu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x2 = (xindex // 262144)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7ltn66jv5rfnapihnadrphqgjifaz4owqfozlfhg7m4rykacjr.py
# Source Nodes: [x_30], Original ATen: [aten.convolution]
# x_30 => convolution_7
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
# x_31 => add_26, add_27, add_28, mul_42, mul_43, mul_44, mul_45, mul_46, rsqrt_5, squeeze_16, var_mean_5
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


# kernel path: /tmp/torchinductor_youkaichao/aq/caqckktnuyfvamrmm6eoggsappnhe5cv2bgwosbycbfifaqlkptu.py
# Source Nodes: [shortcut_1, x_31, x_39, x_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut_1 => mul_55, sigmoid_6
# x_31 => add_26, add_29, mul_41, mul_47, rsqrt_5, sub_5, var_mean_5
# x_39 => add_31, add_34, mul_48, mul_54, rsqrt_6, sub_6, var_mean_6
# x_43 => add_35
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = tmp26 * tmp27
    tmp29 = 1.0
    tmp30 = tmp29 - tmp27
    tmp31 = tmp26 * tmp30
    tmp32 = tmp31 + tmp29
    tmp33 = tmp27 * tmp32
    tl.store(out_ptr1 + (x2), tmp28, None)
    tl.store(out_ptr2 + (x2), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/76/c76obxokdixfmbtef5jdgctscd66bn2kb2u5l5bdima3g4527mbd.py
# Source Nodes: [shortcut_2, x_59, x_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut_2 => mul_80, sigmoid_10
# x_59 => add_47, add_50, mul_73, mul_79, rsqrt_9, sub_9, var_mean_9
# x_66 => add_51
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 1.0
    tmp19 = tmp18 - tmp16
    tmp20 = tmp15 * tmp19
    tmp21 = tmp20 + tmp18
    tmp22 = tmp16 * tmp21
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxf54cc3zhskblsdtuhbc6md7gwybsmbncua6kzdcjzm2glhtj4.py
# Source Nodes: [x_67], Original ATen: [aten.convolution]
# x_67 => convolution_14
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
# x_68 => add_53, add_54, add_55, mul_82, mul_83, mul_84, mul_85, mul_86, rsqrt_10, squeeze_31, var_mean_10
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


# kernel path: /tmp/torchinductor_youkaichao/n5/cn5nj52kwaezfpondxwf2z5bunfnwembc3a7vngi7qd55whsccgr.py
# Source Nodes: [x_68, x_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_68 => add_53, add_56, mul_81, mul_87, rsqrt_10, sub_10, var_mean_10
# x_72 => mul_88, sigmoid_11
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pz3e4d5ozq74xuwttwbtyxt53acqms6xx7kqe3cwhdzltyumhd.py
# Source Nodes: [x_73], Original ATen: [aten.convolution]
# x_73 => convolution_15
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
# x_74 => add_58, add_59, add_60, mul_90, mul_91, mul_92, mul_93, mul_94, rsqrt_11, squeeze_34, var_mean_11
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


# kernel path: /tmp/torchinductor_youkaichao/rz/crzg5wj7jnsbhcyg6oqpqcz3sgsjo7zqddq2vdiuyjyicxdys27n.py
# Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
# x_74 => add_58, add_61, mul_89, mul_95, rsqrt_11, sub_11, var_mean_11
triton_poi_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceavvmri4mw2ju54c7azoh3unsokxwmcm566macoghjhy5s4r7ds.py
# Source Nodes: [x_78, x_se_8], Original ATen: [aten.mean, aten.silu]
# x_78 => mul_96, sigmoid_12
# x_se_8 => mean_2
triton_red_fused_mean_silu_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkdiigdeecxtg4kt3ojwgm324tjoqsmqo2mvhvgpdyv7jnxp4jk.py
# Source Nodes: [x_78, x_se_8], Original ATen: [aten.mean, aten.silu]
# x_78 => mul_96, sigmoid_12
# x_se_8 => mean_2
triton_per_fused_mean_silu_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_44', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (1024*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 1024.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmo73ucj72gqwsy75yncbmmf4k537pvtt4gz6c7wbcmhk6utsw5p.py
# Source Nodes: [x_se_11], Original ATen: [aten.convolution]
# x_se_11 => convolution_17
triton_poi_fused_convolution_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4w/c4wub4ztznmiz66kvfc56pqrr6y7rrydqy737naq3xpdehbijpsl.py
# Source Nodes: [sigmoid_2, x_78, x_80], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# sigmoid_2 => sigmoid_13
# x_78 => mul_96, sigmoid_12
# x_80 => mul_97
triton_poi_fused_mul_sigmoid_silu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = (xindex // 131072)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (128*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbyjobu6bhaehodulyg2fhl65l6nzzt6rem3i32w7mddml4jxqsw.py
# Source Nodes: [x_81], Original ATen: [aten.convolution]
# x_81 => convolution_18
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


# kernel path: /tmp/torchinductor_youkaichao/fl/cflivuvnddf2nmx4jchv4hbhexjqx3kluxobwpiebsresznb27cw.py
# Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
# x_82 => var_mean_12
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


# kernel path: /tmp/torchinductor_youkaichao/k6/ck6ru22pipm3hzg6x7fon5wprj4dhxk7zwpreclfqljgscdpmwjg.py
# Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
# x_82 => add_63, add_64, add_65, mul_100, mul_101, mul_102, mul_103, mul_99, rsqrt_12, squeeze_37, var_mean_12
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_49', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/yt/cytmltpekgl46tn5m36bqwfkyxqb5h2sjdi7ngyas6rth6fb5khy.py
# Source Nodes: [shortcut_3, x_82, x_90, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut_3 => mul_112, sigmoid_14
# x_82 => add_63, add_66, mul_104, mul_98, rsqrt_12, sub_12, var_mean_12
# x_90 => add_68, add_71, mul_105, mul_111, rsqrt_13, sub_13, var_mean_13
# x_94 => add_72
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = tmp26 * tmp27
    tmp29 = 1.0
    tmp30 = tmp29 - tmp27
    tmp31 = tmp26 * tmp30
    tmp32 = tmp31 + tmp29
    tmp33 = tmp27 * tmp32
    tl.store(out_ptr1 + (x2), tmp28, None)
    tl.store(out_ptr2 + (x2), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7277275robueskysgmnommrtg3pp2xjiyz2hies4ic6o42rarx.py
# Source Nodes: [x_100, x_96], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_100 => mul_120, sigmoid_15
# x_96 => add_74, add_77, mul_113, mul_119, rsqrt_14, sub_14, var_mean_14
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr234beafrstyv4v4wnlvfst65xwiuv42elknfdpesncihrzinod.py
# Source Nodes: [shortcut_4, x_110, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut_4 => mul_137, sigmoid_18
# x_110 => add_84, add_87, mul_130, mul_136, rsqrt_16, sub_16, var_mean_16
# x_117 => add_88
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 1.0
    tmp19 = tmp18 - tmp16
    tmp20 = tmp15 * tmp19
    tmp21 = tmp20 + tmp18
    tmp22 = tmp16 * tmp21
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahvo6anivjjxw35pdqrz3v7jjzy6h36jqybuilyt3vzy7vepgdp.py
# Source Nodes: [reshape], Original ATen: [aten.clone]
# reshape => clone_16
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_53', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ip/cipikwofi5ierd2vutmwx6ppjk7pscv3luo6hypz4j2rskhuissj.py
# Source Nodes: [k_1], Original ATen: [aten.clone]
# k_1 => clone_17
triton_poi_fused_clone_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_54', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ps/cps4ohp6g7vujchemzreo4znvukw774256cecfs23tycksfecm47.py
# Source Nodes: [x_126], Original ATen: [aten._unsafe_view, aten.clone]
# x_126 => clone_19, view_7
triton_poi_fused__unsafe_view_clone_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((32*((((32*((y0 // 32) % 32)) + (y0 % 32)) // 32) % 32)) + (1024*((((32*((y0 // 32) % 32)) + (1024*x1) + (32768*(y0 // 1024)) + (y0 % 32)) // 1024) % 1024)) + (y0 % 32)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (32*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r7/cr7vumjgpnazax4r3i3f55cf6sa4na4up5kxcmemcjwq7qagso5v.py
# Source Nodes: [x_130], Original ATen: [aten._unsafe_view, aten.clone]
# x_130 => clone_20, view_13
triton_poi_fused__unsafe_view_clone_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*((((32*(x1 % 32)) + ((x1 // 32) % 32)) // 32) % 32)) + (1024*((((32*(x1 % 32)) + (1024*x0) + (32768*(x1 // 1024)) + ((x1 // 32) % 32)) // 1024) % 1024)) + ((x1 // 32) % 32)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3ai5bnfwjiatyvio2pxkt6xggs4rywcxaxunm2ow45mkkjbb3p6.py
# Source Nodes: [attn, attn_1, mul_4], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn => add_95
# attn_1 => amax, div, exp, sub_18, sum_1
# mul_4 => mul_146
triton_red_fused__softmax_add_mul_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.1767766952966369
        tmp2 = tmp0 * tmp1
        tmp3 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp4 = tl.full([1, 1], 2048, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp7 = tl.full([1, 1], 63, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp5, tmp12, tmp13)
        tmp15 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp16 = tmp15 < tmp4
        tmp17 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp18 = tmp17 < tmp7
        tmp19 = tmp18 & tmp16
        tmp20 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
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
        tmp30 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = 0.1767766952966369
        tmp32 = tmp30 * tmp31
        tmp33 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp34 = tl.full([1, 1], 2048, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp37 = tl.full([1, 1], 63, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp39, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
        tmp42 = tl.where(tmp39, tmp40, tmp41)
        tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
        tmp44 = tl.where(tmp35, tmp42, tmp43)
        tmp45 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp46 = tmp45 < tmp34
        tmp47 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp48 = tmp47 < tmp37
        tmp49 = tmp48 & tmp46
        tmp50 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp49, eviction_policy='evict_last', other=0.0)
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
        tmp62 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = 0.1767766952966369
        tmp64 = tmp62 * tmp63
        tmp65 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp66 = tl.full([1, 1], 2048, tl.int64)
        tmp67 = tmp65 < tmp66
        tmp68 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp69 = tl.full([1, 1], 63, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tmp70 & tmp67
        tmp72 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp71, tmp72, tmp73)
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp67, tmp74, tmp75)
        tmp77 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp78 = tmp77 < tmp66
        tmp79 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp80 = tmp79 < tmp69
        tmp81 = tmp80 & tmp78
        tmp82 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
        tmp84 = tl.where(tmp81, tmp82, tmp83)
        tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
        tmp86 = tl.where(tmp78, tmp84, tmp85)
        tmp87 = tmp76 + tmp86
        tmp88 = tmp64 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl.exp(tmp89)
        tmp91 = tmp90 / tmp60
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp91, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jh/cjhbasagt3rxktkp6e6z57s6yuyd7r73effyhj74ndchhh35el7x.py
# Source Nodes: [reshape_2], Original ATen: [aten.clone]
# reshape_2 => clone_18
triton_poi_fused_clone_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_58', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/id/cidgtclfg23exd3uxrebxoju2x224gy6uhkkzwmjxwhgljhvsuxe.py
# Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
# x_135 => var_mean_18
triton_red_fused__native_batch_norm_legit_functional_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_59', 'mutated_arg_names': []}
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
        tmp0 = tl.load(in_ptr0 + ((32*(((32*(((r2 + (128*x1)) // 32) % 32)) + (r2 % 32)) % 1024)) + (32768*((((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (131072*((r2 + (128*x1)) // 1024)) + (r2 % 32)) // 32768) % 32)) + ((((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (r2 % 32)) // 1024) % 32)), rmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ci/cciubuoafepdch6u6idi5hchwczz57mfv26avbi66elrkl5kcjf3.py
# Source Nodes: [x_135, x_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_135 => add_100, add_97, mul_147, mul_153, rsqrt_18, sub_19, var_mean_18
# x_138 => mul_154, sigmoid_20
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 1024
    x2 = (xindex // 131072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (32768*((x1 + (1024*x0)) // 32768)) + (131072*x2) + (x0 % 32)), None)
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2w/c2woiugi3zigdbge63nlzzdqmxyr7z4wkkersm3lkkseogu6w3lo.py
# Source Nodes: [x_147], Original ATen: [aten.convolution]
# x_147 => convolution_28
triton_poi_fused_convolution_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_61', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vy/cvykhg4u7b45wwqy2ghz4hlv6wchgcfiygg75b7j6qsckfxs4zid.py
# Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_functional]
# x_148 => var_mean_20
triton_red_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5vdzpzljzpquc4kvl7qohzic7wf5bwcrqw5u2loevqzyua5mj3.py
# Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_functional]
# x_148 => add_108, add_109, add_110, mul_164, mul_165, mul_166, mul_167, mul_168, rsqrt_20, squeeze_61, var_mean_20
triton_per_fused__native_batch_norm_legit_functional_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_63', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ky/ckybo2z7hzb64ztgi5qlxdutd2iq43osbp6vrarm2mtg2jquengx.py
# Source Nodes: [x_148, x_152], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_148 => add_108, add_111, mul_163, mul_169, rsqrt_20, sub_21, var_mean_20
# x_152 => mul_170, sigmoid_22
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p3/cp33k4sgmctp4atoxlnsswsiacawf3b65ec36zp6qghvcbv77t7j.py
# Source Nodes: [x_153], Original ATen: [aten.convolution]
# x_153 => convolution_29
triton_poi_fused_convolution_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_65', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ii/ciihxrtefczodqpgfkknetexompjzmzvibvofipiiatmhwppwqnr.py
# Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_functional]
# x_154 => var_mean_21
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


# kernel path: /tmp/torchinductor_youkaichao/it/cit3fymud2vqo3yp3hkhwaomq3ysygly7mmsuwedbdiatdnlizmg.py
# Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_functional]
# x_154 => add_113, add_114, add_115, mul_172, mul_173, mul_174, mul_175, mul_176, rsqrt_21, squeeze_64, var_mean_21
triton_per_fused__native_batch_norm_legit_functional_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_67', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/c6/cc67lybexvynhth33xt2cqc2cmqswhy253swyl5aq5xp2zhqwrv6.py
# Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_functional]
# x_154 => add_113, add_116, mul_171, mul_177, rsqrt_21, sub_22, var_mean_21
triton_poi_fused__native_batch_norm_legit_functional_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_68', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7indsbe4ttrppa35pojn662kssroa7diq6nxksgjucam2tdrat.py
# Source Nodes: [x_158, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_158 => mul_178, sigmoid_23
# x_se_16 => mean_4
triton_red_fused_mean_silu_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3jsesztswgpgtwoaxts3crnk3gbmm5wrkivsb536vxe24p2wcy.py
# Source Nodes: [x_158, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_158 => mul_178, sigmoid_23
# x_se_16 => mean_4
triton_per_fused_mean_silu_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_70', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (512*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 256.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxwcf3vo4augk63ojlotit7w5qcn3wkned2vempi2xo6s7qou6xn.py
# Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.relu]
# x_se_17 => convolution_30
# x_se_18 => relu_4
triton_poi_fused_convolution_relu_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/et/cetvnw3mclqwwlshzm7g5roahshinbiqld357zpzdiowckfnazh5.py
# Source Nodes: [x_se_19], Original ATen: [aten.convolution]
# x_se_19 => convolution_31
triton_poi_fused_convolution_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_72', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cquijogvb26lf6wntcywfsl7k5udxxr7aeicrkdyqecx2m55iyx3.py
# Source Nodes: [sigmoid_4, x_158, x_160], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# sigmoid_4 => sigmoid_24
# x_158 => mul_178, sigmoid_23
# x_160 => mul_179
triton_poi_fused_mul_sigmoid_silu_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 65536)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbpf7stej3ywr44e7fs2e2csdlytcnfn5roisitsybzxsa6emkh.py
# Source Nodes: [x_161], Original ATen: [aten.convolution]
# x_161 => convolution_32
triton_poi_fused_convolution_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_74', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7n/c7n77wdprdugvehpfa47bzdqfnkup2wojhksuwn7sfby6u3s3wnl.py
# Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_functional]
# x_162 => var_mean_22
triton_red_fused__native_batch_norm_legit_functional_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_75', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qe/cqevkfz6yja7mtxtpqglgt7qkisnnjz2pdkzxmpnw6nlnd4xqhpx.py
# Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_functional]
# x_162 => add_118, add_119, add_120, mul_181, mul_182, mul_183, mul_184, mul_185, rsqrt_22, squeeze_67, var_mean_22
triton_per_fused__native_batch_norm_legit_functional_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_76', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/h2/ch22h6l6dcrrkwfihseowjie57fl7eh5daj4jqpbv7pwha73xog5.py
# Source Nodes: [shortcut_6, x_162, x_170, x_174], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut_6 => mul_194, sigmoid_25
# x_162 => add_118, add_121, mul_180, mul_186, rsqrt_22, sub_23, var_mean_22
# x_170 => add_123, add_126, mul_187, mul_193, rsqrt_23, sub_24, var_mean_23
# x_174 => add_127
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = tmp26 * tmp27
    tmp29 = 1.0
    tmp30 = tmp29 - tmp27
    tmp31 = tmp26 * tmp30
    tmp32 = tmp31 + tmp29
    tmp33 = tmp27 * tmp32
    tl.store(out_ptr1 + (x2), tmp28, None)
    tl.store(out_ptr2 + (x2), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdymcixxmmn6fs7crhjz3ax7k66673wkaljzk4bhvn2gk2x47zjj.py
# Source Nodes: [x_176, x_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_176 => add_129, add_132, mul_195, mul_201, rsqrt_24, sub_25, var_mean_24
# x_180 => mul_202, sigmoid_26
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xt/cxtnq6p6ccbebh2detw5szfehduq26gso7cqrjt6xdhma7otqbzc.py
# Source Nodes: [shortcut_7, x_190, x_197], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut_7 => mul_219, sigmoid_29
# x_190 => add_139, add_142, mul_212, mul_218, rsqrt_26, sub_27, var_mean_26
# x_197 => add_143
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 1.0
    tmp19 = tmp18 - tmp16
    tmp20 = tmp15 * tmp19
    tmp21 = tmp20 + tmp18
    tmp22 = tmp16 * tmp21
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rv/crvulgq7ychx4cpv2ssqxlqscb5dznwzdoavtkhfcw5yrrwyzisd.py
# Source Nodes: [reshape_12], Original ATen: [aten.clone]
# reshape_12 => clone_32
triton_poi_fused_clone_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_80', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/x3/cx3hzwnm3dvfcsvvtwsp4hqkdcbzgjgdlqjv4cgog5wacombfgcr.py
# Source Nodes: [k_3], Original ATen: [aten.clone]
# k_3 => clone_33
triton_poi_fused_clone_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_81', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zv/czvlcgnrbpo4lqy2kwxrjjetnsagtrlwctcbrxrky55ahvgbyuw6.py
# Source Nodes: [x_206], Original ATen: [aten._unsafe_view, aten.clone]
# x_206 => clone_35, view_31
triton_poi_fused__unsafe_view_clone_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_82', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtaf4l5otekphegliia36lizu5quywjkhcjnrl2vu53mjlascal.py
# Source Nodes: [x_210], Original ATen: [aten._unsafe_view, aten.clone]
# x_210 => clone_36, view_37
triton_poi_fused__unsafe_view_clone_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_83', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/am/camqop3d4fqelsyngatv6ld2wdjfws27swf63rn73nf7fjlm72cm.py
# Source Nodes: [attn_2, attn_3, mul_7], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_2 => add_150
# attn_3 => amax_1, div_1, exp_1, sub_29, sum_2
# mul_7 => mul_228
triton_red_fused__softmax_add_mul_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_84', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ko/ckow7np43kssyf4pmqrmbuxygiokfbkhigsnb4n5k6waamunqvt7.py
# Source Nodes: [reshape_14], Original ATen: [aten.clone]
# reshape_14 => clone_34
triton_poi_fused_clone_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_85', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4k/c4k6zzjevpmt3ki5w2bvc4fh5kpas2dqo5nwvn6wvy5pz4swbv5f.py
# Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_functional]
# x_215 => var_mean_28
triton_red_fused__native_batch_norm_legit_functional_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtmhlkr67gdu2doarir7q3x3yui2o7mtyyenvylr7kvgjw3rjn6.py
# Source Nodes: [x_215, x_218], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_215 => add_152, add_155, mul_229, mul_235, rsqrt_28, sub_30, var_mean_28
# x_218 => mul_236, sigmoid_31
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuzq3tx3vqk7lk37sddkt3gaaffxk2sndgeorfctvaice3b3r252.py
# Source Nodes: [x_227], Original ATen: [aten.convolution]
# x_227 => convolution_42
triton_poi_fused_convolution_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_88', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vy/cvyj6h7djti35m7evvcsg3rgvqkxpi22p6m4makktgq4j4rycmzq.py
# Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
# x_228 => var_mean_30
triton_red_fused__native_batch_norm_legit_functional_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_89', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/z4/cz4fucgkoq3l6r6gyisgq4cu26ohqodkvix6ccaudotzrpbq2ew7.py
# Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
# x_228 => add_163, add_164, add_165, mul_246, mul_247, mul_248, mul_249, mul_250, rsqrt_30, squeeze_91, var_mean_30
triton_per_fused__native_batch_norm_legit_functional_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_90', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/jv/cjvbgmuby3u5qho3frj3diqlnhhg5oj6rd3ihat6rlgtqr74nely.py
# Source Nodes: [x_228, x_232], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_228 => add_163, add_166, mul_245, mul_251, rsqrt_30, sub_32, var_mean_30
# x_232 => mul_252, sigmoid_33
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcyse3oe2vda53v3wre6utzhm4db2o47ctfhwtyrp2trizk3n5t.py
# Source Nodes: [x_235], Original ATen: [aten._unsafe_view, aten.clone]
# x_235 => clone_45, view_55
triton_poi_fused__unsafe_view_clone_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_92', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/45/c45orw6vls6o7gpdqkr7aeipvmvur2zb5juvzgnqhfvmw7cmduzv.py
# Source Nodes: [x_239], Original ATen: [aten._unsafe_view, aten.clone]
# x_239 => clone_46, view_61
triton_poi_fused__unsafe_view_clone_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_93', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ji/cjic2gyj6rkyi6tfrhadxpvnceol3vpyxbex5lv4iglnig6zar3i.py
# Source Nodes: [attn_4, attn_5, mul_8], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_4 => add_168
# attn_5 => amax_2, div_2, exp_2, sub_33, sum_3
# mul_8 => mul_253
triton_red_fused__softmax_add_mul_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_94', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wy/cwycp7rb6n5xota2ti3m2ofvtvpwzagtjncmto4t2zlox5ays5zh.py
# Source Nodes: [out_4], Original ATen: [aten._unsafe_view, aten.clone]
# out_4 => clone_48, view_71
triton_poi_fused__unsafe_view_clone_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_95', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vv/cvvdpw7q7ub3bs3eucqngywgskapi5yri6gdqr2qkkxt7ezlwlib.py
# Source Nodes: [x_243], Original ATen: [aten.avg_pool2d]
# x_243 => avg_pool2d
triton_poi_fused_avg_pool2d_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_96', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/mu/cmuqwoyru22aq4u7mlxqchpvf2zgd24e25ysxmj7xncsuskkvhmy.py
# Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_functional]
# x_244 => var_mean_31
triton_red_fused__native_batch_norm_legit_functional_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_97', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/36/c36klrhgoo4jbupzvnjo42flbmj23kandmqnshsaokfzpe6jmfcs.py
# Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_functional]
# x_244 => add_170, add_171, add_172, mul_255, mul_256, mul_257, mul_258, mul_259, rsqrt_31, squeeze_94, var_mean_31
triton_per_fused__native_batch_norm_legit_functional_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_98', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/mn/cmnboe2atxs234q5abne2y3ck4kjy5o6d4rojk3vkl63tcete3zd.py
# Source Nodes: [x_244, x_247], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_244 => add_170, add_173, mul_254, mul_260, rsqrt_31, sub_34, var_mean_31
# x_247 => mul_261, sigmoid_34
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_99', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zm/czmlwp6afspxyjnh3vudcd2gf2wec5v6sllswrblaqj7wwcudyek.py
# Source Nodes: [x_248], Original ATen: [aten.convolution]
# x_248 => convolution_44
triton_poi_fused_convolution_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_100', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1536
    y1 = (yindex // 1536)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1536*x2) + (98304*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mg/cmgdjuju6jvxa7hdghfwkeph7xmid3olfjg5lw2zrz4e2lw6y5gd.py
# Source Nodes: [x_249], Original ATen: [aten._native_batch_norm_legit_functional]
# x_249 => var_mean_32
triton_red_fused__native_batch_norm_legit_functional_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1536*r2) + (196608*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mb/cmb2pjkk33bhhi3qi3cmwcbdwuls6s4eeilioun22sbgzfyfltzf.py
# Source Nodes: [x_249], Original ATen: [aten._native_batch_norm_legit_functional]
# x_249 => add_175, add_176, add_177, mul_263, mul_264, mul_265, mul_266, mul_267, rsqrt_32, squeeze_97, var_mean_32
triton_per_fused__native_batch_norm_legit_functional_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_102', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/e5/ce52r7ei5odp6ikp4y3wjlgcg3qvg6n5uuyaz64chrklelrphkhd.py
# Source Nodes: [shortcut_9, x_249, x_256, x_260], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut_9 => mul_276, sigmoid_35
# x_249 => add_175, add_178, mul_262, mul_268, rsqrt_32, sub_35, var_mean_32
# x_256 => add_180, add_183, mul_269, mul_275, rsqrt_33, sub_36, var_mean_33
# x_260 => add_184
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
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
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = tmp26 * tmp27
    tmp29 = 1.0
    tmp30 = tmp29 - tmp27
    tmp31 = tmp26 * tmp30
    tmp32 = tmp31 + tmp29
    tmp33 = tmp27 * tmp32
    tl.store(out_ptr1 + (x2), tmp28, None)
    tl.store(out_ptr2 + (x2), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sr/csriewgr7zimtjrr6xhynn6q4f3uuvsfd4s3uz7awrhbdeaoayfp.py
# Source Nodes: [x_261], Original ATen: [aten.convolution]
# x_261 => convolution_46
triton_poi_fused_convolution_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_104', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/us/cus4wbn7l3rv3o7vduqq5ldsezlhufst54wr4jbk45wgomrw6qhz.py
# Source Nodes: [reshape_36], Original ATen: [aten.clone]
# reshape_36 => clone_52
triton_poi_fused_clone_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_105', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7avspa3xqrajpzswwvcxkacq6j4sr6mcpsp4zkmefl32rlur4d.py
# Source Nodes: [k_7], Original ATen: [aten.clone]
# k_7 => clone_53
triton_poi_fused_clone_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_106', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqqoctzuhjfkbxzqusyeysily3ui3jdivme6mdhz5tpwmofo4vr.py
# Source Nodes: [x_269], Original ATen: [aten._unsafe_view, aten.clone]
# x_269 => clone_55, view_79
triton_poi_fused__unsafe_view_clone_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_107', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/uc/cuc7bwbmtrj2niozba62gistwconcgsxopc6tgmns5b7zt7wzr3t.py
# Source Nodes: [x_273], Original ATen: [aten._unsafe_view, aten.clone]
# x_273 => clone_56, view_85
triton_poi_fused__unsafe_view_clone_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_108', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/iv/civhcli67tyoq6sbdwnblrdbtptxy4f7yvgbg4t7kv2fo7llxsce.py
# Source Nodes: [attn_6, attn_7, mul_9], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_6 => add_191
# attn_7 => amax_3, div_3, exp_3, sub_38, sum_4
# mul_9 => mul_285
triton_per_fused__softmax_add_mul_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_109', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/lc/clcusumahzl23cfwbxpb5nlyitcxq6daluibvtkyg76xrmycjgjj.py
# Source Nodes: [reshape_38], Original ATen: [aten.clone]
# reshape_38 => clone_54
triton_poi_fused_clone_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_110', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/oy/coy4s53iacxolzbkzerqab6ti7xbkcfg3elzf7244fjsx2kjcvhm.py
# Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
# x_278 => var_mean_35
triton_red_fused__native_batch_norm_legit_functional_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_111', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/aw/cawbbfobr4jnqv5ce2zwifqxxld6jnjq55twurif54gzykumnurl.py
# Source Nodes: [x_278, x_281], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_278 => add_193, add_196, mul_286, mul_292, rsqrt_35, sub_39, var_mean_35
# x_281 => mul_293, sigmoid_37
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_112 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_112', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yk/cykepevrxofmjcfwkw5rkahmskpvwdhrs2oz6onbesmizgo2ss67.py
# Source Nodes: [x_283, x_289, x_290], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_283 => add_198, add_201, mul_294, mul_300, rsqrt_36, sub_40, var_mean_36
# x_289 => add_202
# x_290 => mul_301, sigmoid_38
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_113', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 1.0
    tmp19 = tmp18 - tmp16
    tmp20 = tmp15 * tmp19
    tmp21 = tmp20 + tmp18
    tmp22 = tmp16 * tmp21
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7ot2yn5dbqsk4c7c4zcrjhxamfdno4wa5ogbe2npxml4w5y34ag.py
# Source Nodes: [x_291], Original ATen: [aten.convolution]
# x_291 => convolution_49
triton_poi_fused_convolution_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_114', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1280
    y1 = (yindex // 1280)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1280*x2) + (81920*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z5/cz5cvi7l744gdxcsnkmp6l2rgreclbrc4hy7u7vfdwrkdhd4m2qh.py
# Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
# x_292 => var_mean_37
triton_red_fused__native_batch_norm_legit_functional_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_115', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (163840*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/p6/cp6kyj7c3y5mxn47f6qsasrrd3slqylcbb3ehdlwxqzt3gbycyjr.py
# Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
# x_292 => add_204, add_205, add_206, mul_303, mul_304, mul_305, mul_306, mul_307, rsqrt_37, squeeze_112, var_mean_37
triton_per_fused__native_batch_norm_legit_functional_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_116', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/c4/cc4jue2ub37jrojr7hvpuqehli5hu4ch3jbqh7xwyqepkjuvonlj.py
# Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_292 => add_204, add_207, mul_302, mul_308, rsqrt_37, sub_41, var_mean_37
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_117 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_117', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp13 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvlupjyrqpifhdi3zjqpc2h53obn3whg4lavwevrqxc4yytgrrq.py
# Source Nodes: [x_297, x_298, x_300], Original ATen: [aten.mean, aten.silu, aten.view]
# x_297 => mul_309, sigmoid_39
# x_298 => mean_6
# x_300 => view_96
triton_per_fused_mean_silu_view_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_view_118', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (81920*x1)), rmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5crgyu6bheixwzsfqxv4ywqfjvurdc3wyosndoe77idl4wdfif.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_119', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263 = args
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
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (63, 32), (32, 1))
    assert_size_stride(primals_38, (63, 32), (32, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (1024, ), (1, ))
    assert_size_stride(primals_48, (1024, ), (1, ))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_50, (1024, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (1024, ), (1, ))
    assert_size_stride(primals_56, (1024, ), (1, ))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (31, 64), (64, 1))
    assert_size_stride(primals_60, (31, 64), (64, 1))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_63, (1024, ), (1, ))
    assert_size_stride(primals_64, (1024, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (31, 128), (128, 1))
    assert_size_stride(primals_68, (31, 128), (128, 1))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (1536, ), (1, ))
    assert_size_stride(primals_72, (1536, ), (1, ))
    assert_size_stride(primals_73, (1536, ), (1, ))
    assert_size_stride(primals_74, (1536, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_77, (15, 128), (128, 1))
    assert_size_stride(primals_78, (15, 128), (128, 1))
    assert_size_stride(primals_79, (512, ), (1, ))
    assert_size_stride(primals_80, (512, ), (1, ))
    assert_size_stride(primals_81, (1536, ), (1, ))
    assert_size_stride(primals_82, (1536, ), (1, ))
    assert_size_stride(primals_83, (1280, ), (1, ))
    assert_size_stride(primals_84, (1280, ), (1, ))
    assert_size_stride(primals_85, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_86, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_87, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_88, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_89, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_90, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_91, (8, ), (1, ))
    assert_size_stride(primals_92, (64, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_95, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_96, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_97, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_98, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_99, (8, ), (1, ))
    assert_size_stride(primals_100, (64, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_103, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_104, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_105, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_106, (8, ), (1, ))
    assert_size_stride(primals_107, (128, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_110, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_111, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_112, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_113, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_114, (8, ), (1, ))
    assert_size_stride(primals_115, (128, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_118, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_119, (384, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_120, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_121, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_122, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_123, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_124, (16, ), (1, ))
    assert_size_stride(primals_125, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_128, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_129, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_130, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_131, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_132, (16, ), (1, ))
    assert_size_stride(primals_133, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_135, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_136, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_137, (768, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_138, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_139, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_141, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_142, (1536, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_143, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_144, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_145, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_146, (1280, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_147, (1000, 1280), (1280, 1))
    assert_size_stride(primals_148, (1000, ), (1, ))
    assert_size_stride(primals_149, (), ())
    assert_size_stride(primals_150, (24, ), (1, ))
    assert_size_stride(primals_151, (24, ), (1, ))
    assert_size_stride(primals_152, (), ())
    assert_size_stride(primals_153, (32, ), (1, ))
    assert_size_stride(primals_154, (32, ), (1, ))
    assert_size_stride(primals_155, (), ())
    assert_size_stride(primals_156, (64, ), (1, ))
    assert_size_stride(primals_157, (64, ), (1, ))
    assert_size_stride(primals_158, (), ())
    assert_size_stride(primals_159, (64, ), (1, ))
    assert_size_stride(primals_160, (64, ), (1, ))
    assert_size_stride(primals_161, (), ())
    assert_size_stride(primals_162, (64, ), (1, ))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (), ())
    assert_size_stride(primals_165, (256, ), (1, ))
    assert_size_stride(primals_166, (256, ), (1, ))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (256, ), (1, ))
    assert_size_stride(primals_169, (256, ), (1, ))
    assert_size_stride(primals_170, (), ())
    assert_size_stride(primals_171, (64, ), (1, ))
    assert_size_stride(primals_172, (64, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (64, ), (1, ))
    assert_size_stride(primals_175, (64, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (256, ), (1, ))
    assert_size_stride(primals_178, (256, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (512, ), (1, ))
    assert_size_stride(primals_187, (512, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (512, ), (1, ))
    assert_size_stride(primals_190, (512, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (128, ), (1, ))
    assert_size_stride(primals_193, (128, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (128, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (512, ), (1, ))
    assert_size_stride(primals_199, (512, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (128, ), (1, ))
    assert_size_stride(primals_202, (128, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (128, ), (1, ))
    assert_size_stride(primals_205, (128, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (512, ), (1, ))
    assert_size_stride(primals_208, (512, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (256, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (1024, ), (1, ))
    assert_size_stride(primals_217, (1024, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (1024, ), (1, ))
    assert_size_stride(primals_220, (1024, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (256, ), (1, ))
    assert_size_stride(primals_223, (256, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (256, ), (1, ))
    assert_size_stride(primals_226, (256, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_229, (1024, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (256, ), (1, ))
    assert_size_stride(primals_232, (256, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (1024, ), (1, ))
    assert_size_stride(primals_238, (1024, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (512, ), (1, ))
    assert_size_stride(primals_241, (512, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (512, ), (1, ))
    assert_size_stride(primals_244, (512, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (1536, ), (1, ))
    assert_size_stride(primals_247, (1536, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (1536, ), (1, ))
    assert_size_stride(primals_250, (1536, ), (1, ))
    assert_size_stride(primals_251, (), ())
    assert_size_stride(primals_252, (512, ), (1, ))
    assert_size_stride(primals_253, (512, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (1536, ), (1, ))
    assert_size_stride(primals_259, (1536, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (1280, ), (1, ))
    assert_size_stride(primals_262, (1280, ), (1, ))
    assert_size_stride(primals_263, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_85, buf0, 72, 9, grid=grid(72, 9), stream=stream0)
        del primals_85
        buf1 = empty_strided((32, 24, 3, 3), (216, 1, 72, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_86, buf1, 768, 9, grid=grid(768, 9), stream=stream0)
        del primals_86
        buf2 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_87, buf2, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_87
        buf3 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_89, buf3, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_89
        buf4 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_97, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_97
        buf5 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_104, buf5, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_104
        buf6 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_112, buf6, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_112
        buf7 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_122, buf7, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_122
        buf8 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_130, buf8, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_130
        buf9 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_263, buf9, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del primals_263
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 24, 128, 128), (393216, 16384, 128, 1))
        buf11 = empty_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf10, buf11, 192, 16384, grid=grid(192, 16384), stream=stream0)
        buf12 = empty_strided((1, 24, 1, 1, 1024), (24576, 1, 24576, 24576, 24), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((1, 24, 1, 1, 1024), (24576, 1, 24576, 24576, 24), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((1, 24, 1, 1, 1024), (24576, 1, 24576, 24576, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf11, buf12, buf13, buf14, 24576, 128, grid=grid(24576), stream=stream0)
        buf15 = empty_strided((1, 24, 1, 1, 8), (192, 1, 192, 192, 24), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((1, 24, 1, 1, 8), (192, 1, 192, 192, 24), device='cuda', dtype=torch.float32)
        buf17 = empty_strided((1, 24, 1, 1, 8), (192, 1, 192, 192, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf12, buf13, buf14, buf15, buf16, buf17, 192, 128, grid=grid(192), stream=stream0)
        del buf12
        del buf13
        del buf14
        buf18 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf19 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf21 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf15, buf16, buf17, primals_150, primals_151, buf18, buf19, buf21, primals_150, primals_151, 24, 8, grid=grid(24), stream=stream0)
        del buf15
        del buf16
        del buf17
        del primals_150
        del primals_151
        buf23 = reinterpret_tensor(buf10, (8, 24, 128, 128), (393216, 1, 3072, 24), 0); del buf10  # reuse
        buf570 = empty_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_11.run(buf11, buf18, buf19, primals_1, primals_2, buf23, buf570, 3145728, grid=grid(3145728), stream=stream0)
        del buf19
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 32, 128, 128), (524288, 16384, 128, 1))
        buf25 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf24, buf25, 256, 16384, grid=grid(256, 16384), stream=stream0)
        buf26 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        buf27 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        buf28 = empty_strided((1, 32, 1, 1, 1024), (32768, 1, 32768, 32768, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf25, buf26, buf27, buf28, 32768, 128, grid=grid(32768), stream=stream0)
        buf29 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        buf30 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        buf31 = empty_strided((1, 32, 1, 1, 8), (256, 1, 256, 256, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf26, buf27, buf28, buf29, buf30, buf31, 256, 128, grid=grid(256), stream=stream0)
        buf32 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf33 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf35 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf29, buf30, buf31, primals_153, primals_154, buf32, buf33, buf35, primals_153, primals_154, 32, 8, grid=grid(32), stream=stream0)
        del primals_153
        del primals_154
        buf37 = reinterpret_tensor(buf24, (8, 32, 128, 128), (524288, 1, 4096, 32), 0); del buf24  # reuse
        buf569 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_16.run(buf25, buf32, buf33, primals_3, primals_4, buf37, buf569, 4194304, grid=grid(4194304), stream=stream0)
        del buf33
        del primals_4
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf39 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf38, buf39, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf40 = empty_strided((1, 64, 1, 1, 256), (16384, 1, 16384, 16384, 64), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((1, 64, 1, 1, 256), (16384, 1, 16384, 16384, 64), device='cuda', dtype=torch.float32)
        buf42 = empty_strided((1, 64, 1, 1, 256), (16384, 1, 16384, 16384, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf39, buf40, buf41, buf42, 16384, 128, grid=grid(16384), stream=stream0)
        buf43 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf44 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf45 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf40, buf41, buf42, buf43, buf44, buf45, 128, 128, grid=grid(128), stream=stream0)
        buf46 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf47 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf49 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf43, buf44, buf45, primals_156, primals_157, buf46, buf47, buf49, primals_156, primals_157, 64, 2, grid=grid(64), stream=stream0)
        del primals_156
        del primals_157
        buf51 = reinterpret_tensor(buf38, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf38  # reuse
        buf568 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_21.run(buf39, buf46, buf47, primals_5, primals_6, buf51, buf568, 2097152, grid=grid(2097152), stream=stream0)
        del primals_6
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf53 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf52, buf53, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf54 = buf42; del buf42  # reuse
        buf55 = buf41; del buf41  # reuse
        buf56 = buf40; del buf40  # reuse
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf53, buf54, buf55, buf56, 16384, 128, grid=grid(16384), stream=stream0)
        buf57 = buf45; del buf45  # reuse
        buf58 = buf44; del buf44  # reuse
        buf59 = buf43; del buf43  # reuse
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf54, buf55, buf56, buf57, buf58, buf59, 128, 128, grid=grid(128), stream=stream0)
        buf60 = buf47; del buf47  # reuse
        buf61 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf63 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf57, buf58, buf59, primals_159, primals_160, buf60, buf61, buf63, primals_159, primals_160, 64, 2, grid=grid(64), stream=stream0)
        del primals_159
        del primals_160
        buf65 = reinterpret_tensor(buf52, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf52  # reuse
        buf567 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_21.run(buf53, buf60, buf61, primals_7, primals_8, buf65, buf567, 2097152, grid=grid(2097152), stream=stream0)
        del primals_8
        # Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf67 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf66, buf67, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf68 = buf56; del buf56  # reuse
        buf69 = buf55; del buf55  # reuse
        buf70 = buf54; del buf54  # reuse
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf67, buf68, buf69, buf70, 16384, 128, grid=grid(16384), stream=stream0)
        buf71 = buf59; del buf59  # reuse
        buf72 = buf58; del buf58  # reuse
        buf73 = buf57; del buf57  # reuse
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf68, buf69, buf70, buf71, buf72, buf73, 128, 128, grid=grid(128), stream=stream0)
        buf74 = buf61; del buf61  # reuse
        buf75 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf77 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf71, buf72, buf73, primals_162, primals_163, buf74, buf75, buf77, primals_162, primals_163, 64, 2, grid=grid(64), stream=stream0)
        del primals_162
        del primals_163
        buf78 = reinterpret_tensor(buf66, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf66  # reuse
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_22.run(buf67, buf74, buf75, primals_9, primals_10, buf78, 2097152, grid=grid(2097152), stream=stream0)
        del primals_10
        buf79 = reinterpret_tensor(buf70, (8, 64, 1, 1, 32), (2048, 1, 16384, 16384, 64), 0); del buf70  # reuse
        # Source Nodes: [x_27, x_se], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_23.run(buf78, buf79, 16384, 128, grid=grid(16384), stream=stream0)
        buf80 = empty_strided((8, 64, 1, 1), (64, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf81 = reinterpret_tensor(buf80, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf80  # reuse
        # Source Nodes: [x_27, x_se], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_24.run(buf81, buf79, 512, 32, grid=grid(512), stream=stream0)
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 8, 1, 1), (8, 1, 1, 1))
        buf83 = reinterpret_tensor(buf82, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf82  # reuse
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_25.run(buf83, primals_91, 64, grid=grid(64), stream=stream0)
        del primals_91
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 64, 1, 1), (64, 1, 1, 1))
        buf85 = reinterpret_tensor(buf84, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf84  # reuse
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf85, primals_93, 512, grid=grid(512), stream=stream0)
        del primals_93
        buf86 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid, x_27, x_29], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_27.run(buf78, buf85, buf86, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf88 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf87, buf88, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        buf89 = empty_strided((1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf88, buf89, buf90, buf91, 65536, 128, grid=grid(65536), stream=stream0)
        buf92 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        buf93 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        buf94 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf89, buf90, buf91, buf92, buf93, buf94, 512, 128, grid=grid(512), stream=stream0)
        buf95 = reinterpret_tensor(buf31, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf31  # reuse
        buf96 = reinterpret_tensor(buf30, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf30  # reuse
        buf98 = reinterpret_tensor(buf29, (256, ), (1, ), 0); del buf29  # reuse
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf92, buf93, buf94, primals_165, primals_166, buf95, buf96, buf98, primals_165, primals_166, 256, 2, grid=grid(256), stream=stream0)
        del primals_165
        del primals_166
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf51, primals_95, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf100 = reinterpret_tensor(buf87, (8, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf87  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf99, buf100, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        buf101 = buf91; del buf91  # reuse
        buf102 = buf90; del buf90  # reuse
        buf103 = buf89; del buf89  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf100, buf101, buf102, buf103, 65536, 128, grid=grid(65536), stream=stream0)
        buf104 = buf94; del buf94  # reuse
        buf105 = buf93; del buf93  # reuse
        buf106 = buf92; del buf92  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf101, buf102, buf103, buf104, buf105, buf106, 512, 128, grid=grid(512), stream=stream0)
        buf107 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf108 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf110 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf104, buf105, buf106, primals_168, primals_169, buf107, buf108, buf110, primals_168, primals_169, 256, 2, grid=grid(256), stream=stream0)
        del primals_168
        del primals_169
        buf112 = reinterpret_tensor(buf99, (8, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf99  # reuse
        buf566 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_1, x_31, x_39, x_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_32.run(buf88, buf95, buf96, primals_11, primals_12, buf100, buf107, buf108, primals_13, primals_14, buf112, buf566, 8388608, grid=grid(8388608), stream=stream0)
        del primals_12
        del primals_14
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf114 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf113, buf114, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf115 = reinterpret_tensor(buf79, (1, 64, 1, 1, 256), (16384, 1, 16384, 16384, 64), 0); del buf79  # reuse
        buf116 = buf69; del buf69  # reuse
        buf117 = buf68; del buf68  # reuse
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf114, buf115, buf116, buf117, 16384, 128, grid=grid(16384), stream=stream0)
        buf118 = buf73; del buf73  # reuse
        buf119 = buf72; del buf72  # reuse
        buf120 = buf71; del buf71  # reuse
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf115, buf116, buf117, buf118, buf119, buf120, 128, 128, grid=grid(128), stream=stream0)
        buf121 = buf75; del buf75  # reuse
        buf122 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf124 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf118, buf119, buf120, primals_171, primals_172, buf121, buf122, buf124, primals_171, primals_172, 64, 2, grid=grid(64), stream=stream0)
        del primals_171
        del primals_172
        buf126 = reinterpret_tensor(buf113, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf113  # reuse
        buf565 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45, x_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_21.run(buf114, buf121, buf122, primals_15, primals_16, buf126, buf565, 2097152, grid=grid(2097152), stream=stream0)
        del primals_16
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf128 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf127, buf128, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf129 = buf117; del buf117  # reuse
        buf130 = buf116; del buf116  # reuse
        buf131 = buf115; del buf115  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf128, buf129, buf130, buf131, 16384, 128, grid=grid(16384), stream=stream0)
        buf132 = buf120; del buf120  # reuse
        buf133 = buf119; del buf119  # reuse
        buf134 = buf118; del buf118  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf129, buf130, buf131, buf132, buf133, buf134, 128, 128, grid=grid(128), stream=stream0)
        buf135 = buf122; del buf122  # reuse
        buf136 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf138 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf132, buf133, buf134, primals_174, primals_175, buf135, buf136, buf138, primals_174, primals_175, 64, 2, grid=grid(64), stream=stream0)
        del primals_174
        del primals_175
        buf139 = reinterpret_tensor(buf127, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf127  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_22.run(buf128, buf135, buf136, primals_17, primals_18, buf139, 2097152, grid=grid(2097152), stream=stream0)
        del buf136
        del primals_18
        buf140 = reinterpret_tensor(buf131, (8, 64, 1, 1, 32), (2048, 1, 16384, 16384, 64), 0); del buf131  # reuse
        # Source Nodes: [x_55, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_23.run(buf139, buf140, 16384, 128, grid=grid(16384), stream=stream0)
        buf141 = reinterpret_tensor(buf106, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf106  # reuse
        buf142 = reinterpret_tensor(buf141, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf141  # reuse
        # Source Nodes: [x_55, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_24.run(buf142, buf140, 512, 32, grid=grid(512), stream=stream0)
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 8, 1, 1), (8, 1, 1, 1))
        buf144 = reinterpret_tensor(buf143, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf143  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_25.run(buf144, primals_99, 64, grid=grid(64), stream=stream0)
        del primals_99
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 64, 1, 1), (64, 1, 1, 1))
        buf146 = reinterpret_tensor(buf145, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf145  # reuse
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf146, primals_101, 512, grid=grid(512), stream=stream0)
        del primals_101
        buf147 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_1, x_55, x_57], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_27.run(buf139, buf146, buf147, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf149 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_58], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf148, buf149, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        buf150 = buf103; del buf103  # reuse
        buf151 = buf102; del buf102  # reuse
        buf152 = buf101; del buf101  # reuse
        # Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf149, buf150, buf151, buf152, 65536, 128, grid=grid(65536), stream=stream0)
        buf153 = buf105; del buf105  # reuse
        buf154 = buf104; del buf104  # reuse
        buf155 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf150, buf151, buf152, buf153, buf154, buf155, 512, 128, grid=grid(512), stream=stream0)
        del buf150
        del buf151
        del buf152
        buf156 = buf96; del buf96  # reuse
        buf157 = buf108; del buf108  # reuse
        buf159 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf153, buf154, buf155, primals_177, primals_178, buf156, buf157, buf159, primals_177, primals_178, 256, 2, grid=grid(256), stream=stream0)
        del primals_177
        del primals_178
        buf161 = reinterpret_tensor(buf148, (8, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf148  # reuse
        buf564 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_2, x_59, x_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_33.run(buf149, buf156, buf157, primals_19, primals_20, buf112, buf161, buf564, 8388608, grid=grid(8388608), stream=stream0)
        del primals_20
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf163 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf162, buf163, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        buf164 = reinterpret_tensor(buf28, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf28  # reuse
        buf165 = reinterpret_tensor(buf27, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf27  # reuse
        buf166 = reinterpret_tensor(buf26, (1, 128, 1, 1, 256), (32768, 1, 32768, 32768, 128), 0); del buf26  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf163, buf164, buf165, buf166, 32768, 128, grid=grid(32768), stream=stream0)
        buf167 = reinterpret_tensor(buf157, (1, 128, 1, 1, 2), (256, 1, 256, 256, 128), 0); del buf157  # reuse
        buf168 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        buf169 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf164, buf165, buf166, buf167, buf168, buf169, 256, 128, grid=grid(256), stream=stream0)
        buf170 = reinterpret_tensor(buf134, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf134  # reuse
        buf171 = reinterpret_tensor(buf133, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf133  # reuse
        buf173 = reinterpret_tensor(buf132, (128, ), (1, ), 0); del buf132  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf167, buf168, buf169, primals_180, primals_181, buf170, buf171, buf173, primals_180, primals_181, 128, 2, grid=grid(128), stream=stream0)
        del primals_180
        del primals_181
        buf175 = reinterpret_tensor(buf162, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf162  # reuse
        buf563 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68, x_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_38.run(buf163, buf170, buf171, primals_21, primals_22, buf175, buf563, 4194304, grid=grid(4194304), stream=stream0)
        del primals_22
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf177 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf176, buf177, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf178 = empty_strided((1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), device='cuda', dtype=torch.float32)
        buf179 = empty_strided((1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf177, buf178, buf179, buf180, 8192, 128, grid=grid(8192), stream=stream0)
        buf181 = buf171; del buf171  # reuse
        buf182 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf184 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf178, buf179, buf180, primals_183, primals_184, buf181, buf182, buf184, primals_183, primals_184, 128, 64, grid=grid(128), stream=stream0)
        del primals_183
        del primals_184
        buf185 = reinterpret_tensor(buf176, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf176  # reuse
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_42.run(buf177, buf181, buf182, primals_23, primals_24, buf185, 1048576, grid=grid(1048576), stream=stream0)
        del primals_24
        buf186 = reinterpret_tensor(buf180, (8, 128, 1, 1, 8), (1024, 1, 8192, 8192, 128), 0); del buf180  # reuse
        # Source Nodes: [x_78, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_43.run(buf185, buf186, 8192, 128, grid=grid(8192), stream=stream0)
        buf187 = empty_strided((8, 128, 1, 1), (128, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf188 = reinterpret_tensor(buf187, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf187  # reuse
        # Source Nodes: [x_78, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_44.run(buf188, buf186, 1024, 8, grid=grid(1024), stream=stream0)
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 8, 1, 1), (8, 1, 1, 1))
        buf190 = reinterpret_tensor(buf189, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf189  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_25.run(buf190, primals_106, 64, grid=grid(64), stream=stream0)
        del primals_106
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 128, 1, 1), (128, 1, 1, 1))
        buf192 = reinterpret_tensor(buf191, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf191  # reuse
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf192, primals_108, 1024, grid=grid(1024), stream=stream0)
        del primals_108
        buf193 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2, x_78, x_80], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_46.run(buf185, buf192, buf193, 1048576, grid=grid(1048576), stream=stream0)
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf195 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf194, buf195, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        buf196 = reinterpret_tensor(buf166, (1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), 0); del buf166  # reuse
        buf197 = reinterpret_tensor(buf165, (1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), 0); del buf165  # reuse
        buf198 = reinterpret_tensor(buf164, (1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), 0); del buf164  # reuse
        # Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf195, buf196, buf197, buf198, 32768, 128, grid=grid(32768), stream=stream0)
        buf199 = reinterpret_tensor(buf155, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf155  # reuse
        buf200 = reinterpret_tensor(buf154, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf154  # reuse
        buf202 = reinterpret_tensor(buf153, (512, ), (1, ), 0); del buf153  # reuse
        # Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf196, buf197, buf198, primals_186, primals_187, buf199, buf200, buf202, primals_186, primals_187, 512, 64, grid=grid(512), stream=stream0)
        del primals_186
        del primals_187
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf161, primals_110, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf204 = reinterpret_tensor(buf194, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf194  # reuse
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf203, buf204, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        buf205 = buf198; del buf198  # reuse
        buf206 = buf197; del buf197  # reuse
        buf207 = buf196; del buf196  # reuse
        # Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf204, buf205, buf206, buf207, 32768, 128, grid=grid(32768), stream=stream0)
        buf208 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf209 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf211 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf205, buf206, buf207, primals_189, primals_190, buf208, buf209, buf211, primals_189, primals_190, 512, 64, grid=grid(512), stream=stream0)
        del primals_189
        del primals_190
        buf213 = reinterpret_tensor(buf203, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf203  # reuse
        buf562 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_3, x_82, x_90, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_50.run(buf195, buf199, buf200, primals_25, primals_26, buf204, buf208, buf209, primals_27, primals_28, buf213, buf562, 4194304, grid=grid(4194304), stream=stream0)
        del primals_26
        del primals_28
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf215 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf214, buf215, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf216 = reinterpret_tensor(buf186, (1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), 0); del buf186  # reuse
        buf217 = buf179; del buf179  # reuse
        buf218 = buf178; del buf178  # reuse
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf215, buf216, buf217, buf218, 8192, 128, grid=grid(8192), stream=stream0)
        buf219 = buf182; del buf182  # reuse
        buf220 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf222 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf216, buf217, buf218, primals_192, primals_193, buf219, buf220, buf222, primals_192, primals_193, 128, 64, grid=grid(128), stream=stream0)
        del primals_192
        del primals_193
        buf224 = reinterpret_tensor(buf214, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf214  # reuse
        buf561 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100, x_96], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_51.run(buf215, buf219, buf220, primals_29, primals_30, buf224, buf561, 1048576, grid=grid(1048576), stream=stream0)
        del primals_30
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf226 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf225, buf226, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf227 = buf218; del buf218  # reuse
        buf228 = buf217; del buf217  # reuse
        buf229 = buf216; del buf216  # reuse
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf226, buf227, buf228, buf229, 8192, 128, grid=grid(8192), stream=stream0)
        buf230 = buf220; del buf220  # reuse
        buf231 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf233 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf227, buf228, buf229, primals_195, primals_196, buf230, buf231, buf233, primals_195, primals_196, 128, 64, grid=grid(128), stream=stream0)
        del primals_195
        del primals_196
        buf234 = reinterpret_tensor(buf225, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf225  # reuse
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_42.run(buf226, buf230, buf231, primals_31, primals_32, buf234, 1048576, grid=grid(1048576), stream=stream0)
        del primals_32
        buf235 = reinterpret_tensor(buf229, (8, 128, 1, 1, 8), (1024, 1, 8192, 8192, 128), 0); del buf229  # reuse
        # Source Nodes: [x_106, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_43.run(buf234, buf235, 8192, 128, grid=grid(8192), stream=stream0)
        buf236 = empty_strided((8, 128, 1, 1), (128, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf237 = reinterpret_tensor(buf236, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf236  # reuse
        # Source Nodes: [x_106, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_44.run(buf237, buf235, 1024, 8, grid=grid(1024), stream=stream0)
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 8, 1, 1), (8, 1, 1, 1))
        buf239 = reinterpret_tensor(buf238, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf238  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_25.run(buf239, primals_114, 64, grid=grid(64), stream=stream0)
        del primals_114
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (8, 128, 1, 1), (128, 1, 1, 1))
        buf241 = reinterpret_tensor(buf240, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf240  # reuse
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf241, primals_116, 1024, grid=grid(1024), stream=stream0)
        del primals_116
        buf242 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_3, x_106, x_108], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_46.run(buf234, buf241, buf242, 1048576, grid=grid(1048576), stream=stream0)
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf244 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf243, buf244, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        buf245 = buf207; del buf207  # reuse
        buf246 = buf206; del buf206  # reuse
        buf247 = buf205; del buf205  # reuse
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf244, buf245, buf246, buf247, 32768, 128, grid=grid(32768), stream=stream0)
        buf248 = buf209; del buf209  # reuse
        buf249 = buf200; del buf200  # reuse
        buf251 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf245, buf246, buf247, primals_198, primals_199, buf248, buf249, buf251, primals_198, primals_199, 512, 64, grid=grid(512), stream=stream0)
        del primals_198
        del primals_199
        buf253 = reinterpret_tensor(buf243, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf243  # reuse
        buf560 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_4, x_110, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_52.run(buf244, buf248, buf249, primals_33, primals_34, buf213, buf253, buf560, 4194304, grid=grid(4194304), stream=stream0)
        del primals_34
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf255 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf254, buf255, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf256 = reinterpret_tensor(buf235, (1, 128, 1, 1, 64), (8192, 1, 8192, 8192, 128), 0); del buf235  # reuse
        buf257 = buf228; del buf228  # reuse
        buf258 = buf227; del buf227  # reuse
        # Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf255, buf256, buf257, buf258, 8192, 128, grid=grid(8192), stream=stream0)
        buf259 = buf231; del buf231  # reuse
        buf260 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf262 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf256, buf257, buf258, primals_201, primals_202, buf259, buf260, buf262, primals_201, primals_202, 128, 64, grid=grid(128), stream=stream0)
        del primals_201
        del primals_202
        buf264 = reinterpret_tensor(buf254, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf254  # reuse
        buf559 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119, x_123], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_51.run(buf255, buf259, buf260, primals_35, primals_36, buf264, buf559, 1048576, grid=grid(1048576), stream=stream0)
        del primals_36
        # Source Nodes: [x_125], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 384, 32, 32), (393216, 1024, 32, 1))
        buf266 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape], Original ATen: [aten.clone]
        triton_poi_fused_clone_53.run(buf265, buf266, 1048576, grid=grid(1048576), stream=stream0)
        buf267 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_54.run(buf265, buf267, 1048576, grid=grid(1048576), stream=stream0)
        buf268 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf266, (32, 1024, 32), (32768, 1, 1024), 0), reinterpret_tensor(buf267, (32, 32, 1024), (32768, 1024, 1), 0), out=buf268)
        buf269 = empty((32768, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_55.run(buf266, buf269, 32768, 32, grid=grid(32768, 32), stream=stream0)
        buf270 = empty((32768, 63), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.mm]
        extern_kernels.mm(buf269, reinterpret_tensor(primals_37, (32, 63), (1, 32), 0), out=buf270)
        buf271 = empty((32768, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_56.run(buf266, buf271, 1048576, grid=grid(1048576), stream=stream0)
        buf272 = empty((32768, 63), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten.mm]
        extern_kernels.mm(buf271, reinterpret_tensor(primals_38, (32, 63), (1, 32), 0), out=buf272)
        buf275 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1, mul_4], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_57.run(buf268, buf272, buf270, buf275, 32768, 1024, grid=grid(32768), stream=stream0)
        del buf268
        del buf270
        del buf272
        buf276 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_58.run(buf265, buf276, 1048576, grid=grid(1048576), stream=stream0)
        del buf265
        buf277 = empty((32, 1024, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf275, reinterpret_tensor(buf276, (32, 1024, 32), (32768, 1, 1024), 0), out=buf277)
        buf278 = buf258; del buf258  # reuse
        buf279 = buf257; del buf257  # reuse
        buf280 = buf256; del buf256  # reuse
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf277, buf278, buf279, buf280, 8192, 128, grid=grid(8192), stream=stream0)
        buf281 = buf260; del buf260  # reuse
        buf282 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf284 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf278, buf279, buf280, primals_204, primals_205, buf281, buf282, buf284, primals_204, primals_205, 128, 64, grid=grid(128), stream=stream0)
        del primals_204
        del primals_205
        buf286 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        buf558 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135, x_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_60.run(buf277, buf281, buf282, primals_39, primals_40, buf286, buf558, 1048576, grid=grid(1048576), stream=stream0)
        del buf282
        del primals_40
        # Source Nodes: [x_139], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf288 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf287, buf288, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        buf289 = buf247; del buf247  # reuse
        buf290 = buf246; del buf246  # reuse
        buf291 = buf245; del buf245  # reuse
        # Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf288, buf289, buf290, buf291, 32768, 128, grid=grid(32768), stream=stream0)
        buf292 = buf249; del buf249  # reuse
        buf293 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf295 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf289, buf290, buf291, primals_207, primals_208, buf292, buf293, buf295, primals_207, primals_208, 512, 64, grid=grid(512), stream=stream0)
        del buf289
        del buf290
        del buf291
        del primals_207
        del primals_208
        buf297 = reinterpret_tensor(buf287, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf287  # reuse
        buf557 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5, x_140, x_146], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_52.run(buf288, buf292, buf293, primals_41, primals_42, buf253, buf297, buf557, 4194304, grid=grid(4194304), stream=stream0)
        del primals_42
        # Source Nodes: [x_147], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf299 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf298, buf299, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf300 = reinterpret_tensor(buf140, (1, 256, 1, 1, 64), (16384, 1, 16384, 16384, 256), 0); del buf140  # reuse
        buf301 = reinterpret_tensor(buf130, (1, 256, 1, 1, 64), (16384, 1, 16384, 16384, 256), 0); del buf130  # reuse
        buf302 = reinterpret_tensor(buf129, (1, 256, 1, 1, 64), (16384, 1, 16384, 16384, 256), 0); del buf129  # reuse
        # Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf299, buf300, buf301, buf302, 16384, 128, grid=grid(16384), stream=stream0)
        buf303 = reinterpret_tensor(buf169, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf169  # reuse
        buf304 = reinterpret_tensor(buf168, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf168  # reuse
        buf306 = reinterpret_tensor(buf167, (256, ), (1, ), 0); del buf167  # reuse
        # Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf300, buf301, buf302, primals_210, primals_211, buf303, buf304, buf306, primals_210, primals_211, 256, 64, grid=grid(256), stream=stream0)
        del primals_210
        del primals_211
        buf308 = reinterpret_tensor(buf298, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf298  # reuse
        buf556 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148, x_152], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_64.run(buf299, buf303, buf304, primals_43, primals_44, buf308, buf556, 2097152, grid=grid(2097152), stream=stream0)
        del primals_44
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf308, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf310 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf309, buf310, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf311 = empty_strided((1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), device='cuda', dtype=torch.float32)
        buf312 = empty_strided((1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), device='cuda', dtype=torch.float32)
        buf313 = empty_strided((1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf310, buf311, buf312, buf313, 4096, 128, grid=grid(4096), stream=stream0)
        buf314 = buf304; del buf304  # reuse
        buf315 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf317 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf311, buf312, buf313, primals_213, primals_214, buf314, buf315, buf317, primals_213, primals_214, 256, 16, grid=grid(256), stream=stream0)
        del primals_213
        del primals_214
        buf318 = reinterpret_tensor(buf309, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf309  # reuse
        # Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_68.run(buf310, buf314, buf315, primals_45, primals_46, buf318, 524288, grid=grid(524288), stream=stream0)
        del primals_46
        buf319 = reinterpret_tensor(buf313, (8, 256, 1, 1, 2), (512, 1, 4096, 4096, 256), 0); del buf313  # reuse
        # Source Nodes: [x_158, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_69.run(buf318, buf319, 4096, 128, grid=grid(4096), stream=stream0)
        buf320 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf321 = reinterpret_tensor(buf320, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf320  # reuse
        # Source Nodes: [x_158, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_70.run(buf321, buf319, 2048, 2, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_123, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (8, 16, 1, 1), (16, 1, 1, 1))
        buf323 = reinterpret_tensor(buf322, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf322  # reuse
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_71.run(buf323, primals_124, 128, grid=grid(128), stream=stream0)
        del primals_124
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, primals_125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (8, 256, 1, 1), (256, 1, 1, 1))
        buf325 = reinterpret_tensor(buf324, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf324  # reuse
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf325, primals_126, 2048, grid=grid(2048), stream=stream0)
        del primals_126
        buf326 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4, x_158, x_160], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_73.run(buf318, buf325, buf326, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf328 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf327, buf328, 8192, 256, grid=grid(8192, 256), stream=stream0)
        buf329 = reinterpret_tensor(buf302, (1, 1024, 1, 1, 16), (16384, 1, 16384, 16384, 1024), 0); del buf302  # reuse
        buf330 = reinterpret_tensor(buf301, (1, 1024, 1, 1, 16), (16384, 1, 16384, 16384, 1024), 0); del buf301  # reuse
        buf331 = reinterpret_tensor(buf300, (1, 1024, 1, 1, 16), (16384, 1, 16384, 16384, 1024), 0); del buf300  # reuse
        # Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf328, buf329, buf330, buf331, 16384, 128, grid=grid(16384), stream=stream0)
        buf332 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf333 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf335 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf329, buf330, buf331, primals_216, primals_217, buf332, buf333, buf335, primals_216, primals_217, 1024, 16, grid=grid(1024), stream=stream0)
        del primals_216
        del primals_217
        # Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf297, primals_128, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf337 = reinterpret_tensor(buf327, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf327  # reuse
        # Source Nodes: [x_169], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf336, buf337, 8192, 256, grid=grid(8192, 256), stream=stream0)
        buf338 = buf331; del buf331  # reuse
        buf339 = buf330; del buf330  # reuse
        buf340 = buf329; del buf329  # reuse
        # Source Nodes: [x_170], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf337, buf338, buf339, buf340, 16384, 128, grid=grid(16384), stream=stream0)
        buf341 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf342 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf344 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf338, buf339, buf340, primals_219, primals_220, buf341, buf342, buf344, primals_219, primals_220, 1024, 16, grid=grid(1024), stream=stream0)
        del primals_219
        del primals_220
        buf346 = reinterpret_tensor(buf336, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf336  # reuse
        buf555 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_6, x_162, x_170, x_174], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_77.run(buf328, buf332, buf333, primals_47, primals_48, buf337, buf341, buf342, primals_49, primals_50, buf346, buf555, 2097152, grid=grid(2097152), stream=stream0)
        del primals_48
        del primals_50
        # Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_129, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf348 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf347, buf348, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf349 = reinterpret_tensor(buf319, (1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), 0); del buf319  # reuse
        buf350 = buf312; del buf312  # reuse
        buf351 = buf311; del buf311  # reuse
        # Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf348, buf349, buf350, buf351, 4096, 128, grid=grid(4096), stream=stream0)
        buf352 = buf315; del buf315  # reuse
        buf353 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf355 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf349, buf350, buf351, primals_222, primals_223, buf352, buf353, buf355, primals_222, primals_223, 256, 16, grid=grid(256), stream=stream0)
        del primals_222
        del primals_223
        buf357 = reinterpret_tensor(buf347, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf347  # reuse
        buf554 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_176, x_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_78.run(buf348, buf352, buf353, primals_51, primals_52, buf357, buf554, 524288, grid=grid(524288), stream=stream0)
        del primals_52
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf359 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf358, buf359, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf360 = buf351; del buf351  # reuse
        buf361 = buf350; del buf350  # reuse
        buf362 = buf349; del buf349  # reuse
        # Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf359, buf360, buf361, buf362, 4096, 128, grid=grid(4096), stream=stream0)
        buf363 = buf353; del buf353  # reuse
        buf364 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf366 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf360, buf361, buf362, primals_225, primals_226, buf363, buf364, buf366, primals_225, primals_226, 256, 16, grid=grid(256), stream=stream0)
        del primals_225
        del primals_226
        buf367 = reinterpret_tensor(buf358, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf358  # reuse
        # Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_68.run(buf359, buf363, buf364, primals_53, primals_54, buf367, 524288, grid=grid(524288), stream=stream0)
        del primals_54
        buf368 = reinterpret_tensor(buf362, (8, 256, 1, 1, 2), (512, 1, 4096, 4096, 256), 0); del buf362  # reuse
        # Source Nodes: [x_186, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_69.run(buf367, buf368, 4096, 128, grid=grid(4096), stream=stream0)
        buf369 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf370 = reinterpret_tensor(buf369, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf369  # reuse
        # Source Nodes: [x_186, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_70.run(buf370, buf368, 2048, 2, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (8, 16, 1, 1), (16, 1, 1, 1))
        buf372 = reinterpret_tensor(buf371, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf371  # reuse
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_71.run(buf372, primals_132, 128, grid=grid(128), stream=stream0)
        del primals_132
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (8, 256, 1, 1), (256, 1, 1, 1))
        buf374 = reinterpret_tensor(buf373, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf373  # reuse
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf374, primals_134, 2048, grid=grid(2048), stream=stream0)
        del primals_134
        buf375 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5, x_186, x_188], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_73.run(buf367, buf374, buf375, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf377 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf376, buf377, 8192, 256, grid=grid(8192, 256), stream=stream0)
        buf378 = buf340; del buf340  # reuse
        buf379 = buf339; del buf339  # reuse
        buf380 = buf338; del buf338  # reuse
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf377, buf378, buf379, buf380, 16384, 128, grid=grid(16384), stream=stream0)
        buf381 = buf342; del buf342  # reuse
        buf382 = buf333; del buf333  # reuse
        buf384 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf378, buf379, buf380, primals_228, primals_229, buf381, buf382, buf384, primals_228, primals_229, 1024, 16, grid=grid(1024), stream=stream0)
        del primals_228
        del primals_229
        buf386 = reinterpret_tensor(buf376, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf376  # reuse
        buf553 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_7, x_190, x_197], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_79.run(buf377, buf381, buf382, primals_55, primals_56, buf346, buf386, buf553, 2097152, grid=grid(2097152), stream=stream0)
        del primals_56
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf387 = extern_kernels.convolution(buf386, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf388 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf387, buf388, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf389 = reinterpret_tensor(buf368, (1, 256, 1, 1, 16), (4096, 1, 4096, 4096, 256), 0); del buf368  # reuse
        buf390 = buf361; del buf361  # reuse
        buf391 = buf360; del buf360  # reuse
        # Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf388, buf389, buf390, buf391, 4096, 128, grid=grid(4096), stream=stream0)
        buf392 = buf364; del buf364  # reuse
        buf393 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf395 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf389, buf390, buf391, primals_231, primals_232, buf392, buf393, buf395, primals_231, primals_232, 256, 16, grid=grid(256), stream=stream0)
        del primals_231
        del primals_232
        buf397 = reinterpret_tensor(buf387, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf387  # reuse
        buf552 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199, x_203], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_78.run(buf388, buf392, buf393, primals_57, primals_58, buf397, buf552, 524288, grid=grid(524288), stream=stream0)
        del primals_58
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf398 = extern_kernels.convolution(buf397, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (8, 768, 16, 16), (196608, 256, 16, 1))
        buf399 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_80.run(buf398, buf399, 524288, grid=grid(524288), stream=stream0)
        buf400 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_81.run(buf398, buf400, 524288, grid=grid(524288), stream=stream0)
        buf401 = empty((32, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (32, 256, 64), (16384, 1, 256), 0), reinterpret_tensor(buf400, (32, 64, 256), (16384, 256, 1), 0), out=buf401)
        buf402 = empty((8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_82.run(buf399, buf402, 8192, 64, grid=grid(8192, 64), stream=stream0)
        buf403 = empty((8192, 31), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten.mm]
        extern_kernels.mm(buf402, reinterpret_tensor(primals_59, (64, 31), (1, 64), 0), out=buf403)
        buf404 = empty((8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_83.run(buf399, buf404, 524288, grid=grid(524288), stream=stream0)
        buf405 = empty((8192, 31), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.mm]
        extern_kernels.mm(buf404, reinterpret_tensor(primals_60, (64, 31), (1, 64), 0), out=buf405)
        buf408 = empty((32, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_2, attn_3, mul_7], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_84.run(buf401, buf405, buf403, buf408, 8192, 256, grid=grid(8192), stream=stream0)
        buf409 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_85.run(buf398, buf409, 524288, grid=grid(524288), stream=stream0)
        del buf398
        buf410 = empty((32, 256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf408, reinterpret_tensor(buf409, (32, 256, 64), (16384, 1, 256), 0), out=buf410)
        buf411 = buf391; del buf391  # reuse
        buf412 = buf390; del buf390  # reuse
        buf413 = buf389; del buf389  # reuse
        # Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf410, buf411, buf412, buf413, 4096, 128, grid=grid(4096), stream=stream0)
        buf414 = buf393; del buf393  # reuse
        buf415 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf417 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf411, buf412, buf413, primals_234, primals_235, buf414, buf415, buf417, primals_234, primals_235, 256, 16, grid=grid(256), stream=stream0)
        del buf411
        del buf412
        del buf413
        del primals_234
        del primals_235
        buf419 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        buf551 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215, x_218], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_87.run(buf410, buf414, buf415, primals_61, primals_62, buf419, buf551, 524288, grid=grid(524288), stream=stream0)
        del buf415
        del primals_62
        # Source Nodes: [x_219], Original ATen: [aten.convolution]
        buf420 = extern_kernels.convolution(buf419, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf421 = reinterpret_tensor(buf401, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf401  # reuse
        # Source Nodes: [x_219], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf420, buf421, 8192, 256, grid=grid(8192, 256), stream=stream0)
        buf422 = buf380; del buf380  # reuse
        buf423 = buf379; del buf379  # reuse
        buf424 = buf378; del buf378  # reuse
        # Source Nodes: [x_220], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf421, buf422, buf423, buf424, 16384, 128, grid=grid(16384), stream=stream0)
        buf425 = buf382; del buf382  # reuse
        buf426 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf428 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_220], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf422, buf423, buf424, primals_237, primals_238, buf425, buf426, buf428, primals_237, primals_238, 1024, 16, grid=grid(1024), stream=stream0)
        del buf422
        del buf423
        del buf424
        del primals_237
        del primals_238
        buf430 = reinterpret_tensor(buf420, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf420  # reuse
        buf550 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_8, x_220, x_226], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_79.run(buf421, buf425, buf426, primals_63, primals_64, buf386, buf430, buf550, 2097152, grid=grid(2097152), stream=stream0)
        del buf426
        del primals_64
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf432 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf431, buf432, 4096, 256, grid=grid(4096, 256), stream=stream0)
        buf433 = reinterpret_tensor(buf280, (1, 512, 1, 1, 16), (8192, 1, 8192, 8192, 512), 0); del buf280  # reuse
        buf434 = reinterpret_tensor(buf279, (1, 512, 1, 1, 16), (8192, 1, 8192, 8192, 512), 0); del buf279  # reuse
        buf435 = reinterpret_tensor(buf278, (1, 512, 1, 1, 16), (8192, 1, 8192, 8192, 512), 0); del buf278  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_89.run(buf432, buf433, buf434, buf435, 8192, 128, grid=grid(8192), stream=stream0)
        buf436 = buf293; del buf293  # reuse
        buf437 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf439 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_90.run(buf433, buf434, buf435, primals_240, primals_241, buf436, buf437, buf439, primals_240, primals_241, 512, 16, grid=grid(512), stream=stream0)
        del buf433
        del buf434
        del buf435
        del primals_240
        del primals_241
        buf441 = reinterpret_tensor(buf431, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf431  # reuse
        buf549 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228, x_232], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_91.run(buf432, buf436, buf437, primals_65, primals_66, buf441, buf549, 1048576, grid=grid(1048576), stream=stream0)
        del primals_66
        # Source Nodes: [x_234], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(buf441, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (8, 1536, 16, 16), (393216, 256, 16, 1))
        buf443 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_53.run(buf442, buf443, 1048576, grid=grid(1048576), stream=stream0)
        buf444 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_54.run(buf442, buf444, 1048576, grid=grid(1048576), stream=stream0)
        buf445 = empty((32, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf443, (32, 256, 128), (32768, 1, 256), 0), reinterpret_tensor(buf444, (32, 128, 256), (32768, 256, 1), 0), out=buf445)
        buf446 = empty((8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_235], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_92.run(buf443, buf446, 8192, 128, grid=grid(8192, 128), stream=stream0)
        buf447 = buf405; del buf405  # reuse
        # Source Nodes: [x_235], Original ATen: [aten.mm]
        extern_kernels.mm(buf446, reinterpret_tensor(primals_67, (128, 31), (1, 128), 0), out=buf447)
        buf448 = empty((8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_93.run(buf443, buf448, 1048576, grid=grid(1048576), stream=stream0)
        buf449 = buf403; del buf403  # reuse
        # Source Nodes: [x_239], Original ATen: [aten.mm]
        extern_kernels.mm(buf448, reinterpret_tensor(primals_68, (128, 31), (1, 128), 0), out=buf449)
        buf452 = empty((32, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_4, attn_5, mul_8], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_94.run(buf445, buf449, buf447, buf452, 8192, 256, grid=grid(8192), stream=stream0)
        del buf445
        del buf447
        del buf449
        buf453 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_58.run(buf442, buf453, 1048576, grid=grid(1048576), stream=stream0)
        del buf442
        buf454 = empty((32, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf452, reinterpret_tensor(buf453, (32, 256, 128), (32768, 1, 256), 0), out=buf454)
        buf455 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_95.run(buf454, buf455, 1048576, grid=grid(1048576), stream=stream0)
        del buf454
        buf456 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_96.run(buf455, buf456, 262144, grid=grid(262144), stream=stream0)
        buf457 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        buf458 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        buf459 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf456, buf457, buf458, buf459, 2048, 128, grid=grid(2048), stream=stream0)
        buf460 = buf437; del buf437  # reuse
        buf461 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf463 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf457, buf458, buf459, primals_243, primals_244, buf460, buf461, buf463, primals_243, primals_244, 512, 4, grid=grid(512), stream=stream0)
        del primals_243
        del primals_244
        buf465 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        buf548 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244, x_247], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_99.run(buf456, buf460, buf461, primals_69, primals_70, buf465, buf548, 262144, grid=grid(262144), stream=stream0)
        del primals_70
        # Source Nodes: [x_248], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(buf465, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (8, 1536, 8, 8), (98304, 64, 8, 1))
        buf467 = empty_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_248], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_100.run(buf466, buf467, 12288, 64, grid=grid(12288, 64), stream=stream0)
        buf468 = empty_strided((1, 1536, 1, 1, 4), (6144, 1, 6144, 6144, 1536), device='cuda', dtype=torch.float32)
        buf469 = empty_strided((1, 1536, 1, 1, 4), (6144, 1, 6144, 6144, 1536), device='cuda', dtype=torch.float32)
        buf470 = empty_strided((1, 1536, 1, 1, 4), (6144, 1, 6144, 6144, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_249], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_101.run(buf467, buf468, buf469, buf470, 6144, 128, grid=grid(6144), stream=stream0)
        buf471 = empty_strided((1, 1536, 1, 1), (1536, 1, 1536, 1536), device='cuda', dtype=torch.float32)
        buf472 = empty_strided((1, 1536, 1, 1), (1536, 1, 1536, 1536), device='cuda', dtype=torch.float32)
        buf474 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_249], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_102.run(buf468, buf469, buf470, primals_246, primals_247, buf471, buf472, buf474, primals_246, primals_247, 1536, 4, grid=grid(1536), stream=stream0)
        del primals_246
        del primals_247
        # Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf430, primals_142, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (8, 1536, 8, 8), (98304, 64, 8, 1))
        buf476 = reinterpret_tensor(buf466, (8, 1536, 8, 8), (98304, 1, 12288, 1536), 0); del buf466  # reuse
        # Source Nodes: [x_255], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_100.run(buf475, buf476, 12288, 64, grid=grid(12288, 64), stream=stream0)
        buf477 = buf470; del buf470  # reuse
        buf478 = buf469; del buf469  # reuse
        buf479 = buf468; del buf468  # reuse
        # Source Nodes: [x_256], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_101.run(buf476, buf477, buf478, buf479, 6144, 128, grid=grid(6144), stream=stream0)
        buf480 = empty_strided((1, 1536, 1, 1), (1536, 1, 1536, 1536), device='cuda', dtype=torch.float32)
        buf481 = empty_strided((1, 1536, 1, 1), (1536, 1, 1536, 1536), device='cuda', dtype=torch.float32)
        buf483 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_102.run(buf477, buf478, buf479, primals_249, primals_250, buf480, buf481, buf483, primals_249, primals_250, 1536, 4, grid=grid(1536), stream=stream0)
        del primals_249
        del primals_250
        buf485 = reinterpret_tensor(buf475, (8, 1536, 8, 8), (98304, 1, 12288, 1536), 0); del buf475  # reuse
        buf547 = empty_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_9, x_249, x_256, x_260], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_103.run(buf467, buf471, buf472, primals_71, primals_72, buf476, buf480, buf481, primals_73, primals_74, buf485, buf547, 786432, grid=grid(786432), stream=stream0)
        del primals_72
        del primals_74
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf485, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf487 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_104.run(buf486, buf487, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf488 = buf459; del buf459  # reuse
        buf489 = buf458; del buf458  # reuse
        buf490 = buf457; del buf457  # reuse
        # Source Nodes: [x_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf487, buf488, buf489, buf490, 2048, 128, grid=grid(2048), stream=stream0)
        buf491 = buf461; del buf461  # reuse
        buf492 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf494 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf488, buf489, buf490, primals_252, primals_253, buf491, buf492, buf494, primals_252, primals_253, 512, 4, grid=grid(512), stream=stream0)
        del primals_252
        del primals_253
        buf496 = reinterpret_tensor(buf486, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf486  # reuse
        buf546 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_262, x_266], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_99.run(buf487, buf491, buf492, primals_75, primals_76, buf496, buf546, 262144, grid=grid(262144), stream=stream0)
        del primals_76
        # Source Nodes: [x_268], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf496, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (8, 1536, 8, 8), (98304, 64, 8, 1))
        buf498 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_105.run(buf497, buf498, 262144, grid=grid(262144), stream=stream0)
        buf499 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_106.run(buf497, buf499, 262144, grid=grid(262144), stream=stream0)
        buf500 = empty((32, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf498, (32, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf499, (32, 128, 64), (8192, 64, 1), 0), out=buf500)
        buf501 = empty((2048, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_107.run(buf498, buf501, 2048, 128, grid=grid(2048, 128), stream=stream0)
        buf502 = empty((2048, 15), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269], Original ATen: [aten.mm]
        extern_kernels.mm(buf501, reinterpret_tensor(primals_77, (128, 15), (1, 128), 0), out=buf502)
        buf503 = empty((2048, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_273], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_108.run(buf498, buf503, 262144, grid=grid(262144), stream=stream0)
        buf504 = empty((2048, 15), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_273], Original ATen: [aten.mm]
        extern_kernels.mm(buf503, reinterpret_tensor(primals_78, (128, 15), (1, 128), 0), out=buf504)
        buf507 = empty((32, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_6, attn_7, mul_9], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_109.run(buf500, buf504, buf502, buf507, 2048, 64, grid=grid(2048), stream=stream0)
        del buf500
        del buf502
        del buf504
        buf508 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_110.run(buf497, buf508, 262144, grid=grid(262144), stream=stream0)
        buf509 = empty((32, 64, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf507, reinterpret_tensor(buf508, (32, 64, 128), (8192, 1, 64), 0), out=buf509)
        buf510 = buf490; del buf490  # reuse
        buf511 = buf489; del buf489  # reuse
        buf512 = buf488; del buf488  # reuse
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_111.run(buf509, buf510, buf511, buf512, 2048, 128, grid=grid(2048), stream=stream0)
        buf513 = buf492; del buf492  # reuse
        buf514 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf516 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf510, buf511, buf512, primals_255, primals_256, buf513, buf514, buf516, primals_255, primals_256, 512, 4, grid=grid(512), stream=stream0)
        del buf510
        del buf511
        del buf512
        del primals_255
        del primals_256
        buf518 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        buf545 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278, x_281], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_112.run(buf509, buf513, buf514, primals_79, primals_80, buf518, buf545, 262144, grid=grid(262144), stream=stream0)
        del buf514
        del primals_80
        # Source Nodes: [x_282], Original ATen: [aten.convolution]
        buf519 = extern_kernels.convolution(buf518, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (8, 1536, 8, 8), (98304, 64, 8, 1))
        buf520 = reinterpret_tensor(buf497, (8, 1536, 8, 8), (98304, 1, 12288, 1536), 0); del buf497  # reuse
        # Source Nodes: [x_282], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_100.run(buf519, buf520, 12288, 64, grid=grid(12288, 64), stream=stream0)
        buf521 = buf479; del buf479  # reuse
        buf522 = buf478; del buf478  # reuse
        buf523 = buf477; del buf477  # reuse
        # Source Nodes: [x_283], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_101.run(buf520, buf521, buf522, buf523, 6144, 128, grid=grid(6144), stream=stream0)
        buf524 = buf481; del buf481  # reuse
        buf525 = buf472; del buf472  # reuse
        buf527 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_283], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_102.run(buf521, buf522, buf523, primals_258, primals_259, buf524, buf525, buf527, primals_258, primals_259, 1536, 4, grid=grid(1536), stream=stream0)
        del buf521
        del buf522
        del buf523
        del primals_258
        del primals_259
        buf529 = reinterpret_tensor(buf519, (8, 1536, 8, 8), (98304, 1, 12288, 1536), 0); del buf519  # reuse
        buf544 = empty_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_283, x_289, x_290], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_113.run(buf520, buf524, buf525, primals_81, primals_82, buf485, buf529, buf544, 786432, grid=grid(786432), stream=stream0)
        del buf525
        del primals_82
        # Source Nodes: [x_291], Original ATen: [aten.convolution]
        buf530 = extern_kernels.convolution(buf529, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (8, 1280, 8, 8), (81920, 64, 8, 1))
        buf531 = empty_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_291], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_114.run(buf530, buf531, 10240, 64, grid=grid(10240, 64), stream=stream0)
        buf532 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        buf533 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        buf534 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_115.run(buf531, buf532, buf533, buf534, 5120, 128, grid=grid(5120), stream=stream0)
        buf535 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.float32)
        buf536 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.float32)
        buf538 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_116.run(buf532, buf533, buf534, primals_261, primals_262, buf535, buf536, buf538, primals_261, primals_262, 1280, 4, grid=grid(1280), stream=stream0)
        del buf532
        del buf533
        del buf534
        del primals_261
        del primals_262
        buf539 = reinterpret_tensor(buf530, (8, 1280, 8, 8), (81920, 1, 10240, 1280), 0); del buf530  # reuse
        buf543 = empty_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_117.run(buf531, buf535, buf536, primals_83, primals_84, buf539, buf543, 655360, grid=grid(655360), stream=stream0)
        del buf536
        del primals_84
        buf540 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf541 = reinterpret_tensor(buf540, (8, 1280), (1280, 1), 0); del buf540  # reuse
        # Source Nodes: [x_297, x_298, x_300], Original ATen: [aten.mean, aten.silu, aten.view]
        triton_per_fused_mean_silu_view_118.run(buf541, buf539, 10240, 64, grid=grid(10240), stream=stream0)
        del buf539
        buf542 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_302], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_148, buf541, reinterpret_tensor(primals_147, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf542)
        del primals_148
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_149, primals_149, 1, grid=grid(1), stream=stream0)
        del primals_149
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_152, primals_152, 1, grid=grid(1), stream=stream0)
        del primals_152
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_155, primals_155, 1, grid=grid(1), stream=stream0)
        del primals_155
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_158, primals_158, 1, grid=grid(1), stream=stream0)
        del primals_158
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_161, primals_161, 1, grid=grid(1), stream=stream0)
        del primals_161
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_164, primals_164, 1, grid=grid(1), stream=stream0)
        del primals_164
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_167, primals_167, 1, grid=grid(1), stream=stream0)
        del primals_167
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_170, primals_170, 1, grid=grid(1), stream=stream0)
        del primals_170
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_173, primals_173, 1, grid=grid(1), stream=stream0)
        del primals_173
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_176, primals_176, 1, grid=grid(1), stream=stream0)
        del primals_176
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_179, primals_179, 1, grid=grid(1), stream=stream0)
        del primals_179
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_182, primals_182, 1, grid=grid(1), stream=stream0)
        del primals_182
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_185, primals_185, 1, grid=grid(1), stream=stream0)
        del primals_185
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_188, primals_188, 1, grid=grid(1), stream=stream0)
        del primals_188
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_191, primals_191, 1, grid=grid(1), stream=stream0)
        del primals_191
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_194, primals_194, 1, grid=grid(1), stream=stream0)
        del primals_194
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_197, primals_197, 1, grid=grid(1), stream=stream0)
        del primals_197
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_200, primals_200, 1, grid=grid(1), stream=stream0)
        del primals_200
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_203, primals_203, 1, grid=grid(1), stream=stream0)
        del primals_203
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_206, primals_206, 1, grid=grid(1), stream=stream0)
        del primals_206
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_209, primals_209, 1, grid=grid(1), stream=stream0)
        del primals_209
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_212, primals_212, 1, grid=grid(1), stream=stream0)
        del primals_212
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_215, primals_215, 1, grid=grid(1), stream=stream0)
        del primals_215
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_218, primals_218, 1, grid=grid(1), stream=stream0)
        del primals_218
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_221, primals_221, 1, grid=grid(1), stream=stream0)
        del primals_221
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_224, primals_224, 1, grid=grid(1), stream=stream0)
        del primals_224
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_227, primals_227, 1, grid=grid(1), stream=stream0)
        del primals_227
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_230, primals_230, 1, grid=grid(1), stream=stream0)
        del primals_230
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_233, primals_233, 1, grid=grid(1), stream=stream0)
        del primals_233
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_236, primals_236, 1, grid=grid(1), stream=stream0)
        del primals_236
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_239, primals_239, 1, grid=grid(1), stream=stream0)
        del primals_239
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_242, primals_242, 1, grid=grid(1), stream=stream0)
        del primals_242
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_245, primals_245, 1, grid=grid(1), stream=stream0)
        del primals_245
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_248, primals_248, 1, grid=grid(1), stream=stream0)
        del primals_248
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_251, primals_251, 1, grid=grid(1), stream=stream0)
        del primals_251
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_254, primals_254, 1, grid=grid(1), stream=stream0)
        del primals_254
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_257, primals_257, 1, grid=grid(1), stream=stream0)
        del primals_257
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(primals_260, primals_260, 1, grid=grid(1), stream=stream0)
        del primals_260
        return (buf542, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_61, primals_63, primals_65, primals_69, primals_71, primals_73, primals_75, primals_79, primals_81, primals_83, buf0, buf1, buf2, primals_88, buf3, primals_90, primals_92, primals_94, primals_95, primals_96, buf4, primals_98, primals_100, primals_102, primals_103, buf5, primals_105, primals_107, primals_109, primals_110, primals_111, buf6, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, buf7, primals_123, primals_125, primals_127, primals_128, primals_129, buf8, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, buf9, buf11, buf21, buf23, buf25, buf35, buf37, buf39, buf49, buf51, buf53, buf63, buf65, buf67, buf77, buf78, buf81, buf83, buf85, buf86, buf88, buf98, buf100, buf110, buf112, buf114, buf124, buf126, buf128, buf138, buf139, buf142, buf144, buf146, buf147, buf149, buf159, buf161, buf163, buf173, buf175, buf177, buf184, buf185, buf188, buf190, buf192, buf193, buf195, buf202, buf204, buf211, buf213, buf215, buf222, buf224, buf226, buf233, buf234, buf237, buf239, buf241, buf242, buf244, buf251, buf253, buf255, buf262, buf264, buf269, buf271, buf277, buf284, buf286, buf288, buf295, buf297, buf299, buf306, buf308, buf310, buf317, buf318, buf321, buf323, buf325, buf326, buf328, buf335, buf337, buf344, buf346, buf348, buf355, buf357, buf359, buf366, buf367, buf370, buf372, buf374, buf375, buf377, buf384, buf386, buf388, buf395, buf397, buf402, buf404, buf410, buf417, buf419, buf421, buf428, buf430, buf432, buf439, buf441, buf446, buf448, buf455, buf456, buf463, buf465, buf467, buf474, buf476, buf483, buf485, buf487, buf494, buf496, buf501, buf503, buf509, buf516, buf518, buf520, buf527, buf529, buf531, buf538, buf541, reinterpret_tensor(primals_147, (1000, 1280), (1280, 1), 0), buf543, reinterpret_tensor(buf535, (1, 1280, 1, 1), (1280, 1, 1, 1), 0), buf544, reinterpret_tensor(buf524, (1, 1536, 1, 1), (1536, 1, 1, 1), 0), buf545, reinterpret_tensor(buf513, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf507, (32, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf508, (32, 128, 64), (8192, 64, 1), 0), buf507, reinterpret_tensor(primals_78, (15, 128), (128, 1), 0), reinterpret_tensor(primals_77, (15, 128), (128, 1), 0), reinterpret_tensor(buf498, (32, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf499, (32, 64, 128), (8192, 1, 64), 0), buf546, reinterpret_tensor(buf491, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf547, reinterpret_tensor(buf480, (1, 1536, 1, 1), (1536, 1, 1, 1), 0), reinterpret_tensor(buf471, (1, 1536, 1, 1), (1536, 1, 1, 1), 0), buf548, reinterpret_tensor(buf460, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf452, (32, 256, 256), (65536, 1, 256), 0), reinterpret_tensor(buf453, (32, 128, 256), (32768, 256, 1), 0), buf452, reinterpret_tensor(primals_68, (31, 128), (128, 1), 0), reinterpret_tensor(primals_67, (31, 128), (128, 1), 0), reinterpret_tensor(buf443, (32, 128, 256), (32768, 256, 1), 0), reinterpret_tensor(buf444, (32, 256, 128), (32768, 1, 256), 0), buf549, reinterpret_tensor(buf436, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf550, reinterpret_tensor(buf425, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf551, reinterpret_tensor(buf414, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf408, (32, 256, 256), (65536, 1, 256), 0), reinterpret_tensor(buf409, (32, 64, 256), (16384, 256, 1), 0), buf408, reinterpret_tensor(primals_60, (31, 64), (64, 1), 0), reinterpret_tensor(primals_59, (31, 64), (64, 1), 0), reinterpret_tensor(buf399, (32, 64, 256), (16384, 256, 1), 0), reinterpret_tensor(buf400, (32, 256, 64), (16384, 1, 256), 0), buf552, reinterpret_tensor(buf392, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf553, reinterpret_tensor(buf381, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf363, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf554, reinterpret_tensor(buf352, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf555, reinterpret_tensor(buf341, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf332, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf314, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf556, reinterpret_tensor(buf303, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf557, reinterpret_tensor(buf292, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf558, reinterpret_tensor(buf281, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf275, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf276, (32, 32, 1024), (32768, 1024, 1), 0), buf275, reinterpret_tensor(primals_38, (63, 32), (32, 1), 0), reinterpret_tensor(primals_37, (63, 32), (32, 1), 0), reinterpret_tensor(buf266, (32, 32, 1024), (32768, 1024, 1), 0), reinterpret_tensor(buf267, (32, 1024, 32), (32768, 1, 1024), 0), buf559, reinterpret_tensor(buf259, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf560, reinterpret_tensor(buf248, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf230, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf561, reinterpret_tensor(buf219, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf562, reinterpret_tensor(buf208, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf199, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf181, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf563, reinterpret_tensor(buf170, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf564, reinterpret_tensor(buf156, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf135, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf565, reinterpret_tensor(buf121, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf566, reinterpret_tensor(buf107, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf95, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf74, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf567, reinterpret_tensor(buf60, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf568, reinterpret_tensor(buf46, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf569, reinterpret_tensor(buf32, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf570, reinterpret_tensor(buf18, (1, 24, 1, 1), (24, 1, 1, 1), 0), )


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
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((63, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((63, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((31, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((31, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((31, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((31, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((15, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((15, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((384, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1536, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1280, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_150 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_153 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_156 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_159 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_162 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_165 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_168 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_171 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_174 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_177 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_186 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_189 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_192 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_198 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_204 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_207 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_219 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_225 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_231 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_237 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_240 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_243 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_246 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_249 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_252 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_255 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_258 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_261 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('sebotnet33ts_256', benchmark_compiled_module)
