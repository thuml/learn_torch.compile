
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


# kernel path: /tmp/torchinductor_youkaichao/m4/cm443q2wp6f6senk2voeij3uopkq6caspjfpcuveh7pv2abkb6p2.py
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
    size_hints=[64, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
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


# kernel path: /tmp/torchinductor_youkaichao/iu/ciusjklo53eb4ijkm5hmcuvqfrtrzhmtf5pfcsitngu6mouzrtqa.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (144*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxh36u5xkbjmd7utjudpw3c7gorvygbkwgsgp2myejb35qs7qys.py
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
    size_hints=[512, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (144*y1)), tmp0, xmask & ymask)
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


# kernel path: /tmp/torchinductor_youkaichao/yu/cyundy7hnx3qvoqrrn7kexcy7fyvr6xvmi6bzkxjitsorqwaohdm.py
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
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (4608*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ow/cowuza6omhjwqiwk73dufeqho4pg2y4ooovjselmkzhdlbho65to.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqin7cqlopqajlcvw7dgymfrxg46b3dvki6mr4ad54oxyhcsfor.py
# Source Nodes: [l__mod___base_layer_0], Original ATen: [aten.convolution]
# l__mod___base_layer_0 => convolution
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 50176
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
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (802816*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbv5iclx7x7gnmvqq5sbuuvtifiwoiqwvizlh4hst3uhgnozkv56.py
# Source Nodes: [l__mod___base_layer_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___base_layer_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2o3mdks74hbr7733re7pcujg6muswrhc7azpeqet235t6clfz2.py
# Source Nodes: [l__mod___base_layer_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___base_layer_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 400
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (16*r2) + (2016*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (16*r2) + (2016*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (16*r2) + (2016*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (16*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (16*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (16*x0)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfacmzzfyf3nb2jtbdh555vx45vicmer3uwnvrxgzhtpr4i3xmt7.py
# Source Nodes: [l__mod___base_layer_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___base_layer_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (16*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 401408.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000024912370735
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


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nkpd3o4tdy2y6q4wqryqepuklahihxvjzzyeqjemfamyqtivor.py
# Source Nodes: [l__mod___base_layer_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___base_layer_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 401408.0
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


# kernel path: /tmp/torchinductor_youkaichao/7i/c7iaijixmo6glcoyqcg6nmoct4cuj7vb7bolcwrlusdsrq2gou2a.py
# Source Nodes: [l__mod___level1_0], Original ATen: [aten.convolution]
# l__mod___level1_0 => convolution_2
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3dqvanzwnrjlvcchsa4vs7765aw3mpgbh6i3lqbyfovsh37pit.py
# Source Nodes: [l__mod___level1_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___level1_1 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qa/cqaaoac6wkl7zzpgpdzewwi3kqw45ynssvd566d3xjsunonf5bam.py
# Source Nodes: [l__mod___level1_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___level1_1 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
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
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (3584*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r2) + (3584*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (32*r2) + (3584*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2gaykiyreixo2efpzbcxli4cc5oo3dkkpz53wvzdj6njnmjqek.py
# Source Nodes: [l__mod___level1_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___level1_1 => add_11, add_12, add_13, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 7
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


# kernel path: /tmp/torchinductor_youkaichao/yr/cyrerx7ev6erhds7s4f7fexs3w7jue7tqdpbkwwcwkovvcyw2jjp.py
# Source Nodes: [l__mod___level1_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___level1_1 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# x_2 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_youkaichao/iw/ciw6c74bdnxehk67b453hk4nh3jvd3gu7ex76jaaowff5breomih.py
# Source Nodes: [bottom], Original ATen: [aten.max_pool2d_with_indices]
# bottom => getitem_6, getitem_7
triton_poi_fused_max_pool2d_with_indices_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 56
    x2 = (xindex // 1792)
    x3 = xindex
    x4 = (xindex // 1792) % 56
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x1) + (7168*x2)), None)
    tmp1 = tl.load(in_ptr0 + (32 + x0 + (64*x1) + (7168*x2)), None)
    tmp3 = tl.load(in_ptr0 + (3584 + x0 + (64*x1) + (7168*x2)), None)
    tmp5 = tl.load(in_ptr0 + (3616 + x0 + (64*x1) + (7168*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = 1 + (2*x1) + (224*x4)
    tmp9 = (2*x1) + (224*x4)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = 112 + (2*x1) + (224*x4)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = 113 + (2*x1) + (224*x4)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wi/cwi5y6ohbwamdejklcizlpsmidgfjn3apoy7rxfszj472d4g6s2x.py
# Source Nodes: [l__mod___level2_project_0], Original ATen: [aten.convolution]
# l__mod___level2_project_0 => convolution_3
triton_poi_fused_convolution_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (401408*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwljnkljjsse6fjln3bz7e3n2wld5epzpa7563gbkn6plbgzc4et.py
# Source Nodes: [shortcut], Original ATen: [aten._native_batch_norm_legit_functional]
# shortcut => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbyqiouqnjxvzyxujnxhyh27cntycv7psonuo3uqqhhtluok4eb.py
# Source Nodes: [shortcut], Original ATen: [aten._native_batch_norm_legit_functional]
# shortcut => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/j7/cj77bgz64bzeyfvaxhrlvsyseslhnrxg2wzl2sqmmy26x4en66z4.py
# Source Nodes: [shortcut], Original ATen: [aten._native_batch_norm_legit_functional]
# shortcut => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/rk/crkcvdf3yfquzfymgkegwzgfvkg6muykcnn7mtqlxlcjrbn5jni6.py
# Source Nodes: [out], Original ATen: [aten.convolution]
# out => convolution_4
triton_poi_fused_convolution_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvqxeehsk7k35rfqgktutxdh753eygm4vu7rgma73uiltkzg5ma.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => var_mean_4
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


# kernel path: /tmp/torchinductor_youkaichao/4i/c4iag33mtwaqjvi7gprbsfzrw7emirdgzs6g6ebqfjio5d4c62k4.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => var_mean_4
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2u/c2ulw4vkwlde73ewarxb7ehudj3coqihe4xebuuplu4vir6a2xbj.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => add_21, add_22, add_23, mul_29, mul_30, mul_31, mul_32, mul_33, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/2t/c2tsl2rsh6za7fpdxv2tszm2f4jiqslrpduflsbrxdxbid24j3ip.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_1 => add_21, add_24, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# out_2 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_27', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kc/ckchuqtogoc6inatbfm7x5t47oawvrkcgjuylsmie7dhjihv4eay.py
# Source Nodes: [out_3], Original ATen: [aten.convolution]
# out_3 => convolution_5
triton_poi_fused_convolution_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co3yx7ft6ua6lzdwn2qgezzi3ka4a5tyz32izgxmehkdyzn5coiu.py
# Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
# out_4 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
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


# kernel path: /tmp/torchinductor_youkaichao/ss/csseewpoig7m6mrgzrmuzkj4bgppjhnhge6b5vwjynpro36ijsgc.py
# Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
# out_4 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (6272*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (6272*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (6272*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2z/c2z4ygiaphprnznpmgdnlwdmvb3r7nmhztkaimbtcna4at2yis2b.py
# Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
# out_4 => add_26, add_27, add_28, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/2m/c2mku36mjg2xixp4bho7mrvxttpvrziij3j56lqtovso33b6uifm.py
# Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_4 => add_26, add_29, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
# out_5 => relu_4
triton_poi_fused__native_batch_norm_legit_functional_relu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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
    tmp4 = 25088.0
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


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3rsghsuyrbhlyx4vbnukewbb7hv74wsyuhcevv4vjs456jredg.py
# Source Nodes: [out_7, out_8, shortcut, shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_7 => add_31, add_34, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# out_8 => add_35
# shortcut => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# shortcut_1 => relu_5
triton_poi_fused__native_batch_norm_legit_functional_add_relu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_youkaichao/a7/ca77cehccuu3iwjdpnipr3g4r7kscmtlzej5cfjrfdwn7n5k4yjs.py
# Source Nodes: [cat_27, out_17, out_18, x2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
# cat_27 => cat
# out_17 => add_47, add_50, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
# out_18 => add_51
# x2 => relu_8
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
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
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x0 + (256*x1)), tmp16, None)
    tl.store(out_ptr1 + (x0 + (256*x1)), tmp14, None)
    tl.store(out_ptr2 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cc/cccrzofnyw2plqr3hnx4sp7k4zgccga4if6txgjtn3r6t4winewk.py
# Source Nodes: [x_4, x_5, x_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# x_4 => add_53, add_56, mul_70, mul_76, rsqrt_10, sub_10, var_mean_10
# x_5 => add_57
# x_8 => relu_9
triton_poi_fused__native_batch_norm_legit_functional_add_relu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0 + (256*x1)), None)
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


# kernel path: /tmp/torchinductor_youkaichao/am/camtf2c6u3k7ntlfhao23fcyy3bzk7cbfozdsep7kczn2l3lsycs.py
# Source Nodes: [bottom_1, cat_23], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
# bottom_1 => getitem_24, getitem_25
# cat_23 => cat_4
triton_poi_fused_cat_max_pool2d_with_indices_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 28
    x2 = (xindex // 3584)
    x3 = xindex
    x4 = (xindex // 3584) % 28
    x6 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (256*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7296 + x0 + (256*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = 1 + (2*x1) + (112*x4)
    tmp9 = (2*x1) + (112*x4)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = 56 + (2*x1) + (112*x4)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = 57 + (2*x1) + (112*x4)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x0 + (1152*x6)), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chxob5vxx33p6kgmwulvtkr7tzrg7hrmj6gosrblptplzepdkryn.py
# Source Nodes: [l__mod___level3_tree1_tree1_project_0], Original ATen: [aten.convolution]
# l__mod___level3_tree1_tree1_project_0 => convolution_11
triton_poi_fused_convolution_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (200704*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceuvflqjag3o65nw4ukwevphfmsxgs55r6jzpmux7jsncitummq4.py
# Source Nodes: [shortcut_4], Original ATen: [aten._native_batch_norm_legit_functional]
# shortcut_4 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
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


# kernel path: /tmp/torchinductor_youkaichao/en/cen5ci7ziqpy36jh3khqcklxf6jxppfetkqivxpzvkzu6kdws7dr.py
# Source Nodes: [shortcut_4], Original ATen: [aten._native_batch_norm_legit_functional]
# shortcut_4 => add_59, add_60, add_61, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/ix/cixye6wexcazjhja7r5nk32cwrulvxdoeo6sp4xpctw6g535rrbz.py
# Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_21 => add_64, add_67, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
# out_22 => relu_10
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
    xnumel = 3211264
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
    tmp4 = 25088.0
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


# kernel path: /tmp/torchinductor_youkaichao/67/c67s5lswkro3apbwwhioe3pf3r3jfj3bhuoyrll5hxfbqhweqfgn.py
# Source Nodes: [out_23], Original ATen: [aten.convolution]
# out_23 => convolution_13
triton_poi_fused_convolution_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (100352*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbbi2setvfti6bqvzlncgorlrjfcacqzqc3mqef6ctey5nw3ewm.py
# Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
# out_24 => var_mean_13
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
    xnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4l/c4lztcc54dstk6oj5sp4kagcgd47fd75p5kgwrhbifydibsaupbg.py
# Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
# out_24 => add_69, add_70, add_71, mul_92, mul_93, mul_94, mul_95, mul_96, rsqrt_13, squeeze_40, var_mean_13
triton_per_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 49
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


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpt6qq4yvykd6ueyhmee63owk7sl5jxpkwqab4fp54q6xat7uoe.py
# Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_24 => add_69, add_72, mul_91, mul_97, rsqrt_13, sub_13, var_mean_13
# out_25 => relu_11
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
    xnumel = 802816
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
    tmp4 = 6272.0
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


# kernel path: /tmp/torchinductor_youkaichao/fk/cfk54c4g6vbbmwlubqdldalbslgqm4jzkypetocbjf5avgo4vi2f.py
# Source Nodes: [out_27, out_28, shortcut_4, shortcut_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_27 => add_74, add_77, mul_104, mul_98, rsqrt_14, sub_14, var_mean_14
# out_28 => add_78
# shortcut_4 => add_59, add_62, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# shortcut_5 => relu_12
triton_poi_fused__native_batch_norm_legit_functional_add_relu_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_youkaichao/ut/cutc6qq7oqvj2yyobtcnlgaj3tsv2fhc7kmptg6r4nrsguu36tfb.py
# Source Nodes: [cat_26, out_37, out_38, x2_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
# cat_26 => cat_1
# out_37 => add_90, add_93, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
# out_38 => add_94
# x2_1 => relu_15
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
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
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x0 + (512*x1)), tmp16, None)
    tl.store(out_ptr1 + (x0 + (512*x1)), tmp14, None)
    tl.store(out_ptr2 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wv/cwv22jg2xa4tkoir42g5wmb7tz3fbm3rwpjs3lnogtvgqedkd3kl.py
# Source Nodes: [x1_2, x_10, x_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# x1_2 => relu_16
# x_10 => add_96, add_99, mul_126, mul_132, rsqrt_18, sub_18, var_mean_18
# x_11 => add_100
triton_poi_fused__native_batch_norm_legit_functional_add_relu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0 + (512*x1)), None)
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


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtgffnwq7ch3sedc35dbjn5x2t2qjdx25qua3jdgxjgmurw7k4d.py
# Source Nodes: [cat_25, out_47, out_48, shortcut_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
# cat_25 => cat_2
# out_47 => add_112, add_115, mul_147, mul_153, rsqrt_21, sub_21, var_mean_21
# out_48 => add_116
# shortcut_7 => relu_19
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
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
    tl.store(out_ptr1 + (x0 + (768*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2k/c2kwsko7qc75wmqcjpiscxdfk5x7p2bajzdrxfzysvyryxt7arcd.py
# Source Nodes: [cat_25, out_57, out_58, x2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
# cat_25 => cat_2
# out_57 => add_128, add_131, mul_168, mul_174, rsqrt_24, sub_24, var_mean_24
# out_58 => add_132
# x2_2 => relu_22
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
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
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x0 + (768*x1)), tmp16, None)
    tl.store(out_ptr1 + (x0 + (768*x1)), tmp14, None)
    tl.store(out_ptr2 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhzksrzk3ilne6hculdaqqeqlml4fuo5n2b66m673df3kxz7aec.py
# Source Nodes: [cat_23, x1_4, x_15, x_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
# cat_23 => cat_4
# x1_4 => relu_23
# x_15 => add_134, add_137, mul_175, mul_181, rsqrt_25, sub_25, var_mean_25
# x_16 => add_138
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0 + (768*x1)), None)
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
    tl.store(out_ptr1 + (x0 + (1152*x1)), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyhhci43pwgw65bkqglv5wncef25y235dvkn46edzw43sk7jiry.py
# Source Nodes: [out_67, out_68, shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_67 => add_150, add_153, mul_196, mul_202, rsqrt_28, sub_28, var_mean_28
# out_68 => add_154
# shortcut_10 => relu_26
triton_poi_fused__native_batch_norm_legit_functional_add_relu_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_youkaichao/3y/c3y5x5y62pliivb2jslmz7j4agqcnjfz6tusn6uzrywjbnaxd7bg.py
# Source Nodes: [cat_23, out_87, out_88, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
# cat_23 => cat_4
# out_87 => add_188, add_191, mul_245, mul_251, rsqrt_35, sub_35, var_mean_35
# out_88 => add_192
# shortcut_12 => relu_33
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
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
    tl.store(out_ptr1 + (x0 + (1152*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvcxc44l53spsn2ckiajszwvzzp63655sfbayndke5d4sw53xyt.py
# Source Nodes: [cat_23, out_97, out_98, x2_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
# cat_23 => cat_4
# out_97 => add_204, add_207, mul_266, mul_272, rsqrt_38, sub_38, var_mean_38
# out_98 => add_208
# x2_4 => relu_36
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
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
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x0 + (1152*x1)), tmp16, None)
    tl.store(out_ptr1 + (x0 + (1152*x1)), tmp14, None)
    tl.store(out_ptr2 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gf/cgfmqi3q6h6ka22vzonl3p6asx3wdvd2hia7vm53kqo7rvn6dcpp.py
# Source Nodes: [x_26, x_27, x_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# x_26 => add_210, add_213, mul_273, mul_279, rsqrt_39, sub_39, var_mean_39
# x_27 => add_214
# x_32 => relu_37
triton_poi_fused__native_batch_norm_legit_functional_add_relu_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0 + (1152*x1)), None)
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


# kernel path: /tmp/torchinductor_youkaichao/qy/cqya7mnq2xyg52g5ayxhk7tw7nppdibsacl45rjzrelbzodq6bzd.py
# Source Nodes: [bottom_8, cat_15], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
# bottom_8 => getitem_88, getitem_89
# cat_15 => cat_12
triton_poi_fused_cat_max_pool2d_with_indices_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 14
    x2 = (xindex // 3584)
    x3 = xindex
    x4 = (xindex // 3584) % 14
    x6 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (512*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7424 + x0 + (512*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = 1 + (2*x1) + (56*x4)
    tmp9 = (2*x1) + (56*x4)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = 28 + (2*x1) + (56*x4)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = 29 + (2*x1) + (56*x4)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x0 + (2816*x6)), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dp/cdpxeg6wzestm547gzfibignfupgkkjiisyzi3erbe4jmo4mdxb2.py
# Source Nodes: [l__mod___level4_tree1_tree1_tree1_project_0], Original ATen: [aten.convolution]
# l__mod___level4_tree1_tree1_tree1_project_0 => convolution_40
triton_poi_fused_convolution_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (100352*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/chugozbi5xscsjoyyduw4dihbu54od3u4jg5zh7fwbn2rp46lfme.py
# Source Nodes: [shortcut_16], Original ATen: [aten._native_batch_norm_legit_functional]
# shortcut_16 => var_mean_40
triton_red_fused__native_batch_norm_legit_functional_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
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
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5j/c5jkqzdammgsoqydhdeb2oa6qzq7x3up7kicfemgphu6c7qgqgnw.py
# Source Nodes: [shortcut_16], Original ATen: [aten._native_batch_norm_legit_functional]
# shortcut_16 => add_216, add_217, add_218, mul_281, mul_282, mul_283, mul_284, mul_285, rsqrt_40, squeeze_121, var_mean_40
triton_per_fused__native_batch_norm_legit_functional_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_58', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/xn/cxn5vtuirch4werynvdngjwsz6fsiycr53bef6jkq3si7sksqdos.py
# Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_101 => add_221, add_224, mul_287, mul_293, rsqrt_41, sub_41, var_mean_41
# out_102 => relu_38
triton_poi_fused__native_batch_norm_legit_functional_relu_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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
    tmp4 = 6272.0
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


# kernel path: /tmp/torchinductor_youkaichao/2e/c2e6hzg6hp6kbh6vblns6q4fzve7m2kjssadp6nxwefgvly7h2ti.py
# Source Nodes: [out_103], Original ATen: [aten.convolution]
# out_103 => convolution_42
triton_poi_fused_convolution_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (50176*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cquilanfpovatmjfb2ic6gf3vtvo6gk25ztmu2foww35vg6q4npc.py
# Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional]
# out_104 => var_mean_42
triton_red_fused__native_batch_norm_legit_functional_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
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
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ox/coxqbbike2dtzsw7tyuu6lbyyypys7zeesbe5yj3fbjjr6xq5y57.py
# Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional]
# out_104 => add_226, add_227, add_228, mul_295, mul_296, mul_297, mul_298, mul_299, rsqrt_42, squeeze_127, var_mean_42
triton_per_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 13
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


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqcoek5fgzkolbdjmdekbpwjbx4acaxd3bib3xicbgjbffzfleg.py
# Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_104 => add_226, add_229, mul_294, mul_300, rsqrt_42, sub_42, var_mean_42
# out_105 => relu_39
triton_poi_fused__native_batch_norm_legit_functional_relu_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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
    tmp4 = 1568.0
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


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqvmquntk5yncazck5qxeptt7x2zqrzkndbnvexlxgzmh7delfj.py
# Source Nodes: [out_107, out_108, shortcut_16, shortcut_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_107 => add_231, add_234, mul_301, mul_307, rsqrt_43, sub_43, var_mean_43
# out_108 => add_235
# shortcut_16 => add_216, add_219, mul_280, mul_286, rsqrt_40, sub_40, var_mean_40
# shortcut_17 => relu_40
triton_poi_fused__native_batch_norm_legit_functional_add_relu_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_64', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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


# kernel path: /tmp/torchinductor_youkaichao/ed/cedc3jvowxrb65asdsyzs6mlaxzo64pqb2awky2c6ykuqx3b5td2.py
# Source Nodes: [cat_22, out_117, out_118, x2_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
# cat_22 => cat_5
# out_117 => add_247, add_250, mul_322, mul_328, rsqrt_46, sub_46, var_mean_46
# out_118 => add_251
# x2_5 => relu_43
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
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
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x0 + (1024*x1)), tmp16, None)
    tl.store(out_ptr1 + (x0 + (1024*x1)), tmp14, None)
    tl.store(out_ptr2 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vv/cvv52ki2dxwqtbe7rhmzvckwuf354isucvwbajidloarj75ue2br.py
# Source Nodes: [x1_9, x_34, x_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# x1_9 => relu_44
# x_34 => add_253, add_256, mul_329, mul_335, rsqrt_47, sub_47, var_mean_47
# x_35 => add_257
triton_poi_fused__native_batch_norm_legit_functional_add_relu_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0 + (1024*x1)), None)
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


# kernel path: /tmp/torchinductor_youkaichao/vg/cvg4sjvsgalpr4ubzbwnseadwbvcyvjxhhosgym6xuff53d5jpm5.py
# Source Nodes: [cat_21, out_127, out_128, shortcut_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
# cat_21 => cat_6
# out_127 => add_269, add_272, mul_350, mul_356, rsqrt_50, sub_50, var_mean_50
# out_128 => add_273
# shortcut_19 => relu_47
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
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
    tl.store(out_ptr1 + (x0 + (1536*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csvczi7v5dbmfteoidn7qzeft5xsdn5vl5zkeyn5vlatbrjila7q.py
# Source Nodes: [cat_21, out_137, out_138, x2_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
# cat_21 => cat_6
# out_137 => add_285, add_288, mul_371, mul_377, rsqrt_53, sub_53, var_mean_53
# out_138 => add_289
# x2_6 => relu_50
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
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
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x0 + (1536*x1)), tmp16, None)
    tl.store(out_ptr1 + (x0 + (1536*x1)), tmp14, None)
    tl.store(out_ptr2 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxvmcny5f4opsmxxzpl55skocq7xwdiwiwionee6c45apbpqark.py
# Source Nodes: [cat_19, x1_11, x_39, x_40], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
# cat_19 => cat_8
# x1_11 => relu_51
# x_39 => add_291, add_294, mul_378, mul_384, rsqrt_54, sub_54, var_mean_54
# x_40 => add_295
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0 + (1536*x1)), None)
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
    tl.store(out_ptr1 + (x0 + (2048*x1)), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csa6fny7yot6tq7sbmowzb4o3l4deqx2j5ounfplhjbmdoquerdr.py
# Source Nodes: [out_147, out_148, shortcut_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_147 => add_307, add_310, mul_399, mul_405, rsqrt_57, sub_57, var_mean_57
# out_148 => add_311
# shortcut_22 => relu_54
triton_poi_fused__native_batch_norm_legit_functional_add_relu_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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


# kernel path: /tmp/torchinductor_youkaichao/iy/ciyhrtaifvtl4hqejf2z6xvurvlxjsjw6uenegg5qkyzvxjx4yem.py
# Source Nodes: [cat_19, out_167, out_168, shortcut_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
# cat_19 => cat_8
# out_167 => add_345, add_348, mul_448, mul_454, rsqrt_64, sub_64, var_mean_64
# out_168 => add_349
# shortcut_24 => relu_61
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
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
    tl.store(out_ptr1 + (x0 + (2048*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/72/c72mq3guhfl4zaejxgzbry4x7auaoglu3k7nrn6b72v5h76rx6du.py
# Source Nodes: [cat_19, out_177, out_178, x2_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
# cat_19 => cat_8
# out_177 => add_361, add_364, mul_469, mul_475, rsqrt_67, sub_67, var_mean_67
# out_178 => add_365
# x2_8 => relu_64
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
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
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x0 + (2048*x1)), tmp16, None)
    tl.store(out_ptr1 + (x0 + (2048*x1)), tmp14, None)
    tl.store(out_ptr2 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wc/cwcxs4udasn4ageqryw6j5bpz4xwly6gmfhnqp24htdwgm4et2ut.py
# Source Nodes: [cat_15, x1_15, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
# cat_15 => cat_12
# x1_15 => relu_65
# x_50 => add_367, add_370, mul_476, mul_482, rsqrt_68, sub_68, var_mean_68
# x_51 => add_371
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0 + (2048*x1)), None)
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
    tl.store(out_ptr1 + (x0 + (2816*x1)), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxth4ypdrdyhdoklqicrwwskjbuioukuky46bncl6rzwsgf6ztj.py
# Source Nodes: [cat_15, x1_19, x_62, x_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
# cat_15 => cat_12
# x1_19 => relu_79
# x_62 => add_443, add_446, mul_574, mul_580, rsqrt_82, sub_82, var_mean_82
# x_63 => add_447
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0 + (1536*x1)), None)
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
    tl.store(out_ptr1 + (x0 + (2816*x1)), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lk/clk5ubdulbtw3znfj6c4wkg3whdw3ozrrdpmc3hoynyrdrlr5mjb.py
# Source Nodes: [cat_15, out_247, out_248, shortcut_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
# cat_15 => cat_12
# out_247 => add_497, add_500, mul_644, mul_650, rsqrt_92, sub_92, var_mean_92
# out_248 => add_501
# shortcut_35 => relu_89
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
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
    tl.store(out_ptr1 + (x0 + (2816*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nu/cnuz3qhpxjyrh6uxflid333bzk3smoxie4oyzfbbt5qxzzymibvu.py
# Source Nodes: [cat_15, out_257, out_258, x2_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
# cat_15 => cat_12
# out_257 => add_513, add_516, mul_665, mul_671, rsqrt_95, sub_95, var_mean_95
# out_258 => add_517
# x2_12 => relu_92
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
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
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x0 + (2816*x1)), tmp16, None)
    tl.store(out_ptr1 + (x0 + (2816*x1)), tmp14, None)
    tl.store(out_ptr2 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4h/c4hbngz7c33rikrh6txiijq2zfsj4twxtrshypm7t2l7xhj6rbp5.py
# Source Nodes: [x_73, x_74, x_80], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# x_73 => add_519, add_522, mul_672, mul_678, rsqrt_96, sub_96, var_mean_96
# x_74 => add_523
# x_80 => relu_93
triton_poi_fused__native_batch_norm_legit_functional_add_relu_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0 + (2816*x1)), None)
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


# kernel path: /tmp/torchinductor_youkaichao/di/cdizcehdu6g6vvefoxada3uvsdzaxg2llpcpp4f3zesgwt46u7xj.py
# Source Nodes: [bottom_23, cat_14], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
# bottom_23 => getitem_210, getitem_211
# cat_14 => cat_13
triton_poi_fused_cat_max_pool2d_with_indices_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584)
    x3 = xindex
    x4 = (xindex // 3584) % 7
    x6 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (1024*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7680 + x0 + (1024*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = 1 + (2*x1) + (28*x4)
    tmp9 = (2*x1) + (28*x4)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = 14 + (2*x1) + (28*x4)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = 15 + (2*x1) + (28*x4)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x0 + (2560*x6)), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/cseyy47ycx4hum3jopsnrtlnxxvfggub372dajio5wmrjsv77aiu.py
# Source Nodes: [l__mod___level5_project_0], Original ATen: [aten.convolution]
# l__mod___level5_project_0 => convolution_97
triton_poi_fused_convolution_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1024*x2) + (50176*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cdaoayahcohdckvhmvpgdkldurkooe45tio2bpbdueybuvxkpbb5.py
# Source Nodes: [shortcut_36], Original ATen: [aten._native_batch_norm_legit_functional]
# shortcut_36 => var_mean_97
triton_red_fused__native_batch_norm_legit_functional_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hf/chfe7ma2ukcihlpl42get4mvwd6goilnvtijzcakkjq7kvljj2v7.py
# Source Nodes: [shortcut_36], Original ATen: [aten._native_batch_norm_legit_functional]
# shortcut_36 => add_525, add_526, add_527, mul_680, mul_681, mul_682, mul_683, mul_684, rsqrt_97, squeeze_292, var_mean_97
triton_per_fused__native_batch_norm_legit_functional_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_81', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxth3xubuvbsyppvljzulzgbybccgtnjljbad5uv3xkz4cs565n.py
# Source Nodes: [out_261, out_262], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_261 => add_530, add_533, mul_686, mul_692, rsqrt_98, sub_98, var_mean_98
# out_262 => relu_94
triton_poi_fused__native_batch_norm_legit_functional_relu_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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
    tmp4 = 1568.0
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


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xzd7zifjvmhaljaxpjh6zz5hkqos6oyx5uwoc3jqih2yxgsif2.py
# Source Nodes: [out_263], Original ATen: [aten.convolution]
# out_263 => convolution_99
triton_poi_fused_convolution_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (25088*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/ciremvajqgiqiwzr4y2k6jhwfitamqx7xw7revxynjmxa7z6n25v.py
# Source Nodes: [out_264], Original ATen: [aten._native_batch_norm_legit_functional]
# out_264 => var_mean_99
triton_red_fused__native_batch_norm_legit_functional_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ca/ccahqygeh7dvg3nnohh6p2teu72il6vhzqieqreqgzgfcmgrxnrx.py
# Source Nodes: [out_264], Original ATen: [aten._native_batch_norm_legit_functional]
# out_264 => add_535, add_536, add_537, mul_694, mul_695, mul_696, mul_697, mul_698, rsqrt_99, squeeze_298, var_mean_99
triton_per_fused__native_batch_norm_legit_functional_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_85', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/gz/cgznvscehcxngkwvxpqesc4qooe5undz2kzjz5azyy43ikvtblfq.py
# Source Nodes: [out_264, out_265], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_264 => add_535, add_538, mul_693, mul_699, rsqrt_99, sub_99, var_mean_99
# out_265 => relu_95
triton_poi_fused__native_batch_norm_legit_functional_relu_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
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
    tmp4 = 392.0
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


# kernel path: /tmp/torchinductor_youkaichao/in/cinfradaak7byuqdm2x2ybtwin65o6a4sctp7fz54d4wcuizrkgr.py
# Source Nodes: [out_267, out_268, shortcut_36, shortcut_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_267 => add_540, add_543, mul_700, mul_706, rsqrt_100, sub_100, var_mean_100
# out_268 => add_544
# shortcut_36 => add_525, add_528, mul_679, mul_685, rsqrt_97, sub_97, var_mean_97
# shortcut_37 => relu_96
triton_poi_fused__native_batch_norm_legit_functional_add_relu_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_87', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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


# kernel path: /tmp/torchinductor_youkaichao/4m/c4mgvqd4btopo2zuruw63d4qdz4gs6gbah5lojyejapr5jh6xz5r.py
# Source Nodes: [cat_14, out_277, out_278, x2_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
# cat_14 => cat_13
# out_277 => add_556, add_559, mul_721, mul_727, rsqrt_103, sub_103, var_mean_103
# out_278 => add_560
# x2_13 => relu_99
triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
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
    tl.store(out_ptr0 + (x0 + (2560*x1)), tmp16, None)
    tl.store(out_ptr1 + (x0 + (2560*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3cfapteb2f4kuil2p2hvjzqnbxxqkabnb3hnrgucxjm4ztlammt.py
# Source Nodes: [x_82, x_83, x_87], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# x_82 => add_562, add_565, mul_728, mul_734, rsqrt_104, sub_104, var_mean_104
# x_83 => add_566
# x_87 => relu_100
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0 + (2560*x1)), None)
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
    tmp19 = tmp14 <= tmp17
    tl.store(out_ptr0 + (x2), tmp16, None)
    tl.store(out_ptr1 + (x2), tmp18, None)
    tl.store(out_ptr2 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinjfp5ltnhuebbrwkkasocerqf4wwmqnoou4cfnoz6wcz3vrcq5.py
# Source Nodes: [x_88], Original ATen: [aten.mean]
# x_88 => mean
triton_per_fused_mean_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_90', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (50176*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5u/c5ud5fek4h2qzrlakwmf3wmcvy577zzkcc6kjjhm5umajq4rinke.py
# Source Nodes: [pred, x_92], Original ATen: [aten.convolution, aten.view]
# pred => view
# x_92 => convolution_105
triton_poi_fused_convolution_view_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_view_91', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a6/ca6lueqxshnufw4gvkw4qt7mjerlpvjbwnqng5tjeqdju4n7plfk.py
# Source Nodes: [l__mod___base_layer_1], Original ATen: [aten.add]
# l__mod___base_layer_1 => add
triton_poi_fused_add_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_92', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_81, (128, ), (1, ))
    assert_size_stride(primals_82, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_92, (128, ), (1, ))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (128, ), (1, ))
    assert_size_stride(primals_103, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_118, (256, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (256, ), (1, ))
    assert_size_stride(primals_121, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_123, (512, ), (1, ))
    assert_size_stride(primals_124, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_132, (512, ), (1, ))
    assert_size_stride(primals_133, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_135, (256, ), (1, ))
    assert_size_stride(primals_136, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (256, ), (1, ))
    assert_size_stride(primals_139, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_140, (512, ), (1, ))
    assert_size_stride(primals_141, (512, ), (1, ))
    assert_size_stride(primals_142, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_147, (256, ), (1, ))
    assert_size_stride(primals_148, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_150, (256, ), (1, ))
    assert_size_stride(primals_151, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_152, (512, ), (1, ))
    assert_size_stride(primals_153, (512, ), (1, ))
    assert_size_stride(primals_154, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_155, (256, ), (1, ))
    assert_size_stride(primals_156, (256, ), (1, ))
    assert_size_stride(primals_157, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_161, (512, ), (1, ))
    assert_size_stride(primals_162, (512, ), (1, ))
    assert_size_stride(primals_163, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_165, (512, ), (1, ))
    assert_size_stride(primals_166, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_167, (256, ), (1, ))
    assert_size_stride(primals_168, (256, ), (1, ))
    assert_size_stride(primals_169, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_170, (256, ), (1, ))
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_174, (512, ), (1, ))
    assert_size_stride(primals_175, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_176, (256, ), (1, ))
    assert_size_stride(primals_177, (256, ), (1, ))
    assert_size_stride(primals_178, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_179, (256, ), (1, ))
    assert_size_stride(primals_180, (256, ), (1, ))
    assert_size_stride(primals_181, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_182, (512, ), (1, ))
    assert_size_stride(primals_183, (512, ), (1, ))
    assert_size_stride(primals_184, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_185, (512, ), (1, ))
    assert_size_stride(primals_186, (512, ), (1, ))
    assert_size_stride(primals_187, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_188, (256, ), (1, ))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_190, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_191, (256, ), (1, ))
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_193, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_197, (256, ), (1, ))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_199, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_201, (256, ), (1, ))
    assert_size_stride(primals_202, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_206, (512, ), (1, ))
    assert_size_stride(primals_207, (512, ), (1, ))
    assert_size_stride(primals_208, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_209, (256, ), (1, ))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_216, (512, ), (1, ))
    assert_size_stride(primals_217, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_220, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_221, (256, ), (1, ))
    assert_size_stride(primals_222, (256, ), (1, ))
    assert_size_stride(primals_223, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_224, (512, ), (1, ))
    assert_size_stride(primals_225, (512, ), (1, ))
    assert_size_stride(primals_226, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_227, (512, ), (1, ))
    assert_size_stride(primals_228, (512, ), (1, ))
    assert_size_stride(primals_229, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_231, (256, ), (1, ))
    assert_size_stride(primals_232, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_236, (512, ), (1, ))
    assert_size_stride(primals_237, (512, ), (1, ))
    assert_size_stride(primals_238, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_239, (256, ), (1, ))
    assert_size_stride(primals_240, (256, ), (1, ))
    assert_size_stride(primals_241, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_244, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_245, (512, ), (1, ))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_248, (512, ), (1, ))
    assert_size_stride(primals_249, (512, ), (1, ))
    assert_size_stride(primals_250, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_252, (256, ), (1, ))
    assert_size_stride(primals_253, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_254, (256, ), (1, ))
    assert_size_stride(primals_255, (256, ), (1, ))
    assert_size_stride(primals_256, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_257, (512, ), (1, ))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (256, ), (1, ))
    assert_size_stride(primals_262, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_264, (256, ), (1, ))
    assert_size_stride(primals_265, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_266, (512, ), (1, ))
    assert_size_stride(primals_267, (512, ), (1, ))
    assert_size_stride(primals_268, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_269, (512, ), (1, ))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_271, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_272, (256, ), (1, ))
    assert_size_stride(primals_273, (256, ), (1, ))
    assert_size_stride(primals_274, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_275, (256, ), (1, ))
    assert_size_stride(primals_276, (256, ), (1, ))
    assert_size_stride(primals_277, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_278, (512, ), (1, ))
    assert_size_stride(primals_279, (512, ), (1, ))
    assert_size_stride(primals_280, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_281, (256, ), (1, ))
    assert_size_stride(primals_282, (256, ), (1, ))
    assert_size_stride(primals_283, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_284, (256, ), (1, ))
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_287, (512, ), (1, ))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_289, (512, 2816, 1, 1), (2816, 1, 1, 1))
    assert_size_stride(primals_290, (512, ), (1, ))
    assert_size_stride(primals_291, (512, ), (1, ))
    assert_size_stride(primals_292, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_293, (1024, ), (1, ))
    assert_size_stride(primals_294, (1024, ), (1, ))
    assert_size_stride(primals_295, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_296, (512, ), (1, ))
    assert_size_stride(primals_297, (512, ), (1, ))
    assert_size_stride(primals_298, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_299, (512, ), (1, ))
    assert_size_stride(primals_300, (512, ), (1, ))
    assert_size_stride(primals_301, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_303, (1024, ), (1, ))
    assert_size_stride(primals_304, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_308, (512, ), (1, ))
    assert_size_stride(primals_309, (512, ), (1, ))
    assert_size_stride(primals_310, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_312, (1024, ), (1, ))
    assert_size_stride(primals_313, (1024, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(primals_314, (1024, ), (1, ))
    assert_size_stride(primals_315, (1024, ), (1, ))
    assert_size_stride(primals_316, (1000, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_317, (1000, ), (1, ))
    assert_size_stride(primals_318, (16, ), (1, ))
    assert_size_stride(primals_319, (16, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (16, ), (1, ))
    assert_size_stride(primals_322, (16, ), (1, ))
    assert_size_stride(primals_323, (), ())
    assert_size_stride(primals_324, (32, ), (1, ))
    assert_size_stride(primals_325, (32, ), (1, ))
    assert_size_stride(primals_326, (), ())
    assert_size_stride(primals_327, (128, ), (1, ))
    assert_size_stride(primals_328, (128, ), (1, ))
    assert_size_stride(primals_329, (), ())
    assert_size_stride(primals_330, (64, ), (1, ))
    assert_size_stride(primals_331, (64, ), (1, ))
    assert_size_stride(primals_332, (), ())
    assert_size_stride(primals_333, (64, ), (1, ))
    assert_size_stride(primals_334, (64, ), (1, ))
    assert_size_stride(primals_335, (), ())
    assert_size_stride(primals_336, (128, ), (1, ))
    assert_size_stride(primals_337, (128, ), (1, ))
    assert_size_stride(primals_338, (), ())
    assert_size_stride(primals_339, (64, ), (1, ))
    assert_size_stride(primals_340, (64, ), (1, ))
    assert_size_stride(primals_341, (), ())
    assert_size_stride(primals_342, (64, ), (1, ))
    assert_size_stride(primals_343, (64, ), (1, ))
    assert_size_stride(primals_344, (), ())
    assert_size_stride(primals_345, (128, ), (1, ))
    assert_size_stride(primals_346, (128, ), (1, ))
    assert_size_stride(primals_347, (), ())
    assert_size_stride(primals_348, (128, ), (1, ))
    assert_size_stride(primals_349, (128, ), (1, ))
    assert_size_stride(primals_350, (), ())
    assert_size_stride(primals_351, (256, ), (1, ))
    assert_size_stride(primals_352, (256, ), (1, ))
    assert_size_stride(primals_353, (), ())
    assert_size_stride(primals_354, (128, ), (1, ))
    assert_size_stride(primals_355, (128, ), (1, ))
    assert_size_stride(primals_356, (), ())
    assert_size_stride(primals_357, (128, ), (1, ))
    assert_size_stride(primals_358, (128, ), (1, ))
    assert_size_stride(primals_359, (), ())
    assert_size_stride(primals_360, (256, ), (1, ))
    assert_size_stride(primals_361, (256, ), (1, ))
    assert_size_stride(primals_362, (), ())
    assert_size_stride(primals_363, (128, ), (1, ))
    assert_size_stride(primals_364, (128, ), (1, ))
    assert_size_stride(primals_365, (), ())
    assert_size_stride(primals_366, (128, ), (1, ))
    assert_size_stride(primals_367, (128, ), (1, ))
    assert_size_stride(primals_368, (), ())
    assert_size_stride(primals_369, (256, ), (1, ))
    assert_size_stride(primals_370, (256, ), (1, ))
    assert_size_stride(primals_371, (), ())
    assert_size_stride(primals_372, (256, ), (1, ))
    assert_size_stride(primals_373, (256, ), (1, ))
    assert_size_stride(primals_374, (), ())
    assert_size_stride(primals_375, (128, ), (1, ))
    assert_size_stride(primals_376, (128, ), (1, ))
    assert_size_stride(primals_377, (), ())
    assert_size_stride(primals_378, (128, ), (1, ))
    assert_size_stride(primals_379, (128, ), (1, ))
    assert_size_stride(primals_380, (), ())
    assert_size_stride(primals_381, (256, ), (1, ))
    assert_size_stride(primals_382, (256, ), (1, ))
    assert_size_stride(primals_383, (), ())
    assert_size_stride(primals_384, (128, ), (1, ))
    assert_size_stride(primals_385, (128, ), (1, ))
    assert_size_stride(primals_386, (), ())
    assert_size_stride(primals_387, (128, ), (1, ))
    assert_size_stride(primals_388, (128, ), (1, ))
    assert_size_stride(primals_389, (), ())
    assert_size_stride(primals_390, (256, ), (1, ))
    assert_size_stride(primals_391, (256, ), (1, ))
    assert_size_stride(primals_392, (), ())
    assert_size_stride(primals_393, (256, ), (1, ))
    assert_size_stride(primals_394, (256, ), (1, ))
    assert_size_stride(primals_395, (), ())
    assert_size_stride(primals_396, (128, ), (1, ))
    assert_size_stride(primals_397, (128, ), (1, ))
    assert_size_stride(primals_398, (), ())
    assert_size_stride(primals_399, (128, ), (1, ))
    assert_size_stride(primals_400, (128, ), (1, ))
    assert_size_stride(primals_401, (), ())
    assert_size_stride(primals_402, (256, ), (1, ))
    assert_size_stride(primals_403, (256, ), (1, ))
    assert_size_stride(primals_404, (), ())
    assert_size_stride(primals_405, (128, ), (1, ))
    assert_size_stride(primals_406, (128, ), (1, ))
    assert_size_stride(primals_407, (), ())
    assert_size_stride(primals_408, (128, ), (1, ))
    assert_size_stride(primals_409, (128, ), (1, ))
    assert_size_stride(primals_410, (), ())
    assert_size_stride(primals_411, (256, ), (1, ))
    assert_size_stride(primals_412, (256, ), (1, ))
    assert_size_stride(primals_413, (), ())
    assert_size_stride(primals_414, (256, ), (1, ))
    assert_size_stride(primals_415, (256, ), (1, ))
    assert_size_stride(primals_416, (), ())
    assert_size_stride(primals_417, (128, ), (1, ))
    assert_size_stride(primals_418, (128, ), (1, ))
    assert_size_stride(primals_419, (), ())
    assert_size_stride(primals_420, (128, ), (1, ))
    assert_size_stride(primals_421, (128, ), (1, ))
    assert_size_stride(primals_422, (), ())
    assert_size_stride(primals_423, (256, ), (1, ))
    assert_size_stride(primals_424, (256, ), (1, ))
    assert_size_stride(primals_425, (), ())
    assert_size_stride(primals_426, (128, ), (1, ))
    assert_size_stride(primals_427, (128, ), (1, ))
    assert_size_stride(primals_428, (), ())
    assert_size_stride(primals_429, (128, ), (1, ))
    assert_size_stride(primals_430, (128, ), (1, ))
    assert_size_stride(primals_431, (), ())
    assert_size_stride(primals_432, (256, ), (1, ))
    assert_size_stride(primals_433, (256, ), (1, ))
    assert_size_stride(primals_434, (), ())
    assert_size_stride(primals_435, (256, ), (1, ))
    assert_size_stride(primals_436, (256, ), (1, ))
    assert_size_stride(primals_437, (), ())
    assert_size_stride(primals_438, (512, ), (1, ))
    assert_size_stride(primals_439, (512, ), (1, ))
    assert_size_stride(primals_440, (), ())
    assert_size_stride(primals_441, (256, ), (1, ))
    assert_size_stride(primals_442, (256, ), (1, ))
    assert_size_stride(primals_443, (), ())
    assert_size_stride(primals_444, (256, ), (1, ))
    assert_size_stride(primals_445, (256, ), (1, ))
    assert_size_stride(primals_446, (), ())
    assert_size_stride(primals_447, (512, ), (1, ))
    assert_size_stride(primals_448, (512, ), (1, ))
    assert_size_stride(primals_449, (), ())
    assert_size_stride(primals_450, (256, ), (1, ))
    assert_size_stride(primals_451, (256, ), (1, ))
    assert_size_stride(primals_452, (), ())
    assert_size_stride(primals_453, (256, ), (1, ))
    assert_size_stride(primals_454, (256, ), (1, ))
    assert_size_stride(primals_455, (), ())
    assert_size_stride(primals_456, (512, ), (1, ))
    assert_size_stride(primals_457, (512, ), (1, ))
    assert_size_stride(primals_458, (), ())
    assert_size_stride(primals_459, (512, ), (1, ))
    assert_size_stride(primals_460, (512, ), (1, ))
    assert_size_stride(primals_461, (), ())
    assert_size_stride(primals_462, (256, ), (1, ))
    assert_size_stride(primals_463, (256, ), (1, ))
    assert_size_stride(primals_464, (), ())
    assert_size_stride(primals_465, (256, ), (1, ))
    assert_size_stride(primals_466, (256, ), (1, ))
    assert_size_stride(primals_467, (), ())
    assert_size_stride(primals_468, (512, ), (1, ))
    assert_size_stride(primals_469, (512, ), (1, ))
    assert_size_stride(primals_470, (), ())
    assert_size_stride(primals_471, (256, ), (1, ))
    assert_size_stride(primals_472, (256, ), (1, ))
    assert_size_stride(primals_473, (), ())
    assert_size_stride(primals_474, (256, ), (1, ))
    assert_size_stride(primals_475, (256, ), (1, ))
    assert_size_stride(primals_476, (), ())
    assert_size_stride(primals_477, (512, ), (1, ))
    assert_size_stride(primals_478, (512, ), (1, ))
    assert_size_stride(primals_479, (), ())
    assert_size_stride(primals_480, (512, ), (1, ))
    assert_size_stride(primals_481, (512, ), (1, ))
    assert_size_stride(primals_482, (), ())
    assert_size_stride(primals_483, (256, ), (1, ))
    assert_size_stride(primals_484, (256, ), (1, ))
    assert_size_stride(primals_485, (), ())
    assert_size_stride(primals_486, (256, ), (1, ))
    assert_size_stride(primals_487, (256, ), (1, ))
    assert_size_stride(primals_488, (), ())
    assert_size_stride(primals_489, (512, ), (1, ))
    assert_size_stride(primals_490, (512, ), (1, ))
    assert_size_stride(primals_491, (), ())
    assert_size_stride(primals_492, (256, ), (1, ))
    assert_size_stride(primals_493, (256, ), (1, ))
    assert_size_stride(primals_494, (), ())
    assert_size_stride(primals_495, (256, ), (1, ))
    assert_size_stride(primals_496, (256, ), (1, ))
    assert_size_stride(primals_497, (), ())
    assert_size_stride(primals_498, (512, ), (1, ))
    assert_size_stride(primals_499, (512, ), (1, ))
    assert_size_stride(primals_500, (), ())
    assert_size_stride(primals_501, (512, ), (1, ))
    assert_size_stride(primals_502, (512, ), (1, ))
    assert_size_stride(primals_503, (), ())
    assert_size_stride(primals_504, (256, ), (1, ))
    assert_size_stride(primals_505, (256, ), (1, ))
    assert_size_stride(primals_506, (), ())
    assert_size_stride(primals_507, (256, ), (1, ))
    assert_size_stride(primals_508, (256, ), (1, ))
    assert_size_stride(primals_509, (), ())
    assert_size_stride(primals_510, (512, ), (1, ))
    assert_size_stride(primals_511, (512, ), (1, ))
    assert_size_stride(primals_512, (), ())
    assert_size_stride(primals_513, (256, ), (1, ))
    assert_size_stride(primals_514, (256, ), (1, ))
    assert_size_stride(primals_515, (), ())
    assert_size_stride(primals_516, (256, ), (1, ))
    assert_size_stride(primals_517, (256, ), (1, ))
    assert_size_stride(primals_518, (), ())
    assert_size_stride(primals_519, (512, ), (1, ))
    assert_size_stride(primals_520, (512, ), (1, ))
    assert_size_stride(primals_521, (), ())
    assert_size_stride(primals_522, (512, ), (1, ))
    assert_size_stride(primals_523, (512, ), (1, ))
    assert_size_stride(primals_524, (), ())
    assert_size_stride(primals_525, (256, ), (1, ))
    assert_size_stride(primals_526, (256, ), (1, ))
    assert_size_stride(primals_527, (), ())
    assert_size_stride(primals_528, (256, ), (1, ))
    assert_size_stride(primals_529, (256, ), (1, ))
    assert_size_stride(primals_530, (), ())
    assert_size_stride(primals_531, (512, ), (1, ))
    assert_size_stride(primals_532, (512, ), (1, ))
    assert_size_stride(primals_533, (), ())
    assert_size_stride(primals_534, (256, ), (1, ))
    assert_size_stride(primals_535, (256, ), (1, ))
    assert_size_stride(primals_536, (), ())
    assert_size_stride(primals_537, (256, ), (1, ))
    assert_size_stride(primals_538, (256, ), (1, ))
    assert_size_stride(primals_539, (), ())
    assert_size_stride(primals_540, (512, ), (1, ))
    assert_size_stride(primals_541, (512, ), (1, ))
    assert_size_stride(primals_542, (), ())
    assert_size_stride(primals_543, (512, ), (1, ))
    assert_size_stride(primals_544, (512, ), (1, ))
    assert_size_stride(primals_545, (), ())
    assert_size_stride(primals_546, (256, ), (1, ))
    assert_size_stride(primals_547, (256, ), (1, ))
    assert_size_stride(primals_548, (), ())
    assert_size_stride(primals_549, (256, ), (1, ))
    assert_size_stride(primals_550, (256, ), (1, ))
    assert_size_stride(primals_551, (), ())
    assert_size_stride(primals_552, (512, ), (1, ))
    assert_size_stride(primals_553, (512, ), (1, ))
    assert_size_stride(primals_554, (), ())
    assert_size_stride(primals_555, (256, ), (1, ))
    assert_size_stride(primals_556, (256, ), (1, ))
    assert_size_stride(primals_557, (), ())
    assert_size_stride(primals_558, (256, ), (1, ))
    assert_size_stride(primals_559, (256, ), (1, ))
    assert_size_stride(primals_560, (), ())
    assert_size_stride(primals_561, (512, ), (1, ))
    assert_size_stride(primals_562, (512, ), (1, ))
    assert_size_stride(primals_563, (), ())
    assert_size_stride(primals_564, (512, ), (1, ))
    assert_size_stride(primals_565, (512, ), (1, ))
    assert_size_stride(primals_566, (), ())
    assert_size_stride(primals_567, (256, ), (1, ))
    assert_size_stride(primals_568, (256, ), (1, ))
    assert_size_stride(primals_569, (), ())
    assert_size_stride(primals_570, (256, ), (1, ))
    assert_size_stride(primals_571, (256, ), (1, ))
    assert_size_stride(primals_572, (), ())
    assert_size_stride(primals_573, (512, ), (1, ))
    assert_size_stride(primals_574, (512, ), (1, ))
    assert_size_stride(primals_575, (), ())
    assert_size_stride(primals_576, (256, ), (1, ))
    assert_size_stride(primals_577, (256, ), (1, ))
    assert_size_stride(primals_578, (), ())
    assert_size_stride(primals_579, (256, ), (1, ))
    assert_size_stride(primals_580, (256, ), (1, ))
    assert_size_stride(primals_581, (), ())
    assert_size_stride(primals_582, (512, ), (1, ))
    assert_size_stride(primals_583, (512, ), (1, ))
    assert_size_stride(primals_584, (), ())
    assert_size_stride(primals_585, (512, ), (1, ))
    assert_size_stride(primals_586, (512, ), (1, ))
    assert_size_stride(primals_587, (), ())
    assert_size_stride(primals_588, (256, ), (1, ))
    assert_size_stride(primals_589, (256, ), (1, ))
    assert_size_stride(primals_590, (), ())
    assert_size_stride(primals_591, (256, ), (1, ))
    assert_size_stride(primals_592, (256, ), (1, ))
    assert_size_stride(primals_593, (), ())
    assert_size_stride(primals_594, (512, ), (1, ))
    assert_size_stride(primals_595, (512, ), (1, ))
    assert_size_stride(primals_596, (), ())
    assert_size_stride(primals_597, (256, ), (1, ))
    assert_size_stride(primals_598, (256, ), (1, ))
    assert_size_stride(primals_599, (), ())
    assert_size_stride(primals_600, (256, ), (1, ))
    assert_size_stride(primals_601, (256, ), (1, ))
    assert_size_stride(primals_602, (), ())
    assert_size_stride(primals_603, (512, ), (1, ))
    assert_size_stride(primals_604, (512, ), (1, ))
    assert_size_stride(primals_605, (), ())
    assert_size_stride(primals_606, (512, ), (1, ))
    assert_size_stride(primals_607, (512, ), (1, ))
    assert_size_stride(primals_608, (), ())
    assert_size_stride(primals_609, (1024, ), (1, ))
    assert_size_stride(primals_610, (1024, ), (1, ))
    assert_size_stride(primals_611, (), ())
    assert_size_stride(primals_612, (512, ), (1, ))
    assert_size_stride(primals_613, (512, ), (1, ))
    assert_size_stride(primals_614, (), ())
    assert_size_stride(primals_615, (512, ), (1, ))
    assert_size_stride(primals_616, (512, ), (1, ))
    assert_size_stride(primals_617, (), ())
    assert_size_stride(primals_618, (1024, ), (1, ))
    assert_size_stride(primals_619, (1024, ), (1, ))
    assert_size_stride(primals_620, (), ())
    assert_size_stride(primals_621, (512, ), (1, ))
    assert_size_stride(primals_622, (512, ), (1, ))
    assert_size_stride(primals_623, (), ())
    assert_size_stride(primals_624, (512, ), (1, ))
    assert_size_stride(primals_625, (512, ), (1, ))
    assert_size_stride(primals_626, (), ())
    assert_size_stride(primals_627, (1024, ), (1, ))
    assert_size_stride(primals_628, (1024, ), (1, ))
    assert_size_stride(primals_629, (), ())
    assert_size_stride(primals_630, (1024, ), (1, ))
    assert_size_stride(primals_631, (1024, ), (1, ))
    assert_size_stride(primals_632, (), ())
    assert_size_stride(primals_633, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((16, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 48, 49, grid=grid(48, 49), stream=stream0)
        del primals_1
        buf1 = empty_strided((16, 16, 3, 3), (144, 1, 48, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_4, buf1, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_4
        buf2 = empty_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_7, buf2, 512, 9, grid=grid(512, 9), stream=stream0)
        del primals_7
        buf3 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_16, buf3, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_16
        buf4 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_25, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_25
        buf5 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_40, buf5, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_40
        buf6 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_49, buf6, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_49
        buf7 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_61, buf7, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_61
        buf8 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_70, buf8, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_70
        buf9 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_82, buf9, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_82
        buf10 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_91, buf10, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_91
        buf11 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_103, buf11, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_103
        buf12 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_112, buf12, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_112
        buf13 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_127, buf13, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_127
        buf14 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_136, buf14, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_136
        buf15 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_148, buf15, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_148
        buf16 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_157, buf16, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_157
        buf17 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_169, buf17, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_169
        buf18 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_178, buf18, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_178
        buf19 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_190, buf19, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_190
        buf20 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_199, buf20, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_199
        buf21 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_211, buf21, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_211
        buf22 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_220, buf22, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_220
        buf23 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_232, buf23, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_232
        buf24 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_241, buf24, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_241
        buf25 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_253, buf25, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_253
        buf26 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_262, buf26, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_262
        buf27 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_274, buf27, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_274
        buf28 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_283, buf28, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_283
        buf29 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_298, buf29, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_298
        buf30 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(primals_307, buf30, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_307
        buf31 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(primals_633, buf31, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_633
        # Source Nodes: [l__mod___base_layer_0], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, buf0, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 16, 224, 224), (802816, 50176, 224, 1))
        buf33 = empty_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___base_layer_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf32, buf33, 128, 50176, grid=grid(128, 50176), stream=stream0)
        buf34 = empty_strided((1, 16, 1, 1, 3136), (50176, 1, 50176, 50176, 16), device='cuda', dtype=torch.float32)
        buf35 = empty_strided((1, 16, 1, 1, 3136), (50176, 1, 50176, 50176, 16), device='cuda', dtype=torch.float32)
        buf36 = empty_strided((1, 16, 1, 1, 3136), (50176, 1, 50176, 50176, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___base_layer_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf33, buf34, buf35, buf36, 50176, 128, grid=grid(50176), stream=stream0)
        buf37 = empty_strided((1, 16, 1, 1, 25), (400, 1, 400, 400, 16), device='cuda', dtype=torch.float32)
        buf38 = empty_strided((1, 16, 1, 1, 25), (400, 1, 400, 400, 16), device='cuda', dtype=torch.float32)
        buf39 = empty_strided((1, 16, 1, 1, 25), (400, 1, 400, 400, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___base_layer_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf34, buf35, buf36, buf37, buf38, buf39, 400, 126, grid=grid(400), stream=stream0)
        buf40 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf43 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___base_layer_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_11.run(buf37, buf38, buf39, primals_318, primals_319, buf40, buf41, buf43, primals_318, primals_319, 16, 25, grid=grid(16), stream=stream0)
        del primals_318
        del primals_319
        buf44 = reinterpret_tensor(buf32, (8, 16, 224, 224), (802816, 1, 3584, 16), 0); del buf32  # reuse
        # Source Nodes: [l__mod___base_layer_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf33, buf40, buf41, primals_2, primals_3, buf44, 6422528, grid=grid(6422528), stream=stream0)
        del primals_3
        # Source Nodes: [l__mod___level0_0], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 16, 224, 224), (802816, 50176, 224, 1))
        buf46 = empty_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf45, buf46, 128, 50176, grid=grid(128, 50176), stream=stream0)
        buf47 = buf36; del buf36  # reuse
        buf48 = buf35; del buf35  # reuse
        buf49 = buf34; del buf34  # reuse
        # Source Nodes: [l__mod___level0_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf46, buf47, buf48, buf49, 50176, 128, grid=grid(50176), stream=stream0)
        buf50 = buf39; del buf39  # reuse
        buf51 = buf38; del buf38  # reuse
        buf52 = buf37; del buf37  # reuse
        # Source Nodes: [l__mod___level0_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf47, buf48, buf49, buf50, buf51, buf52, 400, 126, grid=grid(400), stream=stream0)
        buf53 = buf41; del buf41  # reuse
        buf54 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf56 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level0_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_11.run(buf50, buf51, buf52, primals_321, primals_322, buf53, buf54, buf56, primals_321, primals_322, 16, 25, grid=grid(16), stream=stream0)
        del buf50
        del buf51
        del buf52
        del primals_321
        del primals_322
        buf57 = reinterpret_tensor(buf45, (8, 16, 224, 224), (802816, 1, 3584, 16), 0); del buf45  # reuse
        # Source Nodes: [l__mod___level0_1, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf46, buf53, buf54, primals_5, primals_6, buf57, 6422528, grid=grid(6422528), stream=stream0)
        del buf54
        del primals_6
        # Source Nodes: [l__mod___level1_0], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf59 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf58, buf59, 256, 12544, grid=grid(256, 12544), stream=stream0)
        buf60 = empty_strided((1, 32, 1, 1, 784), (25088, 1, 25088, 25088, 32), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((1, 32, 1, 1, 784), (25088, 1, 25088, 25088, 32), device='cuda', dtype=torch.float32)
        buf62 = empty_strided((1, 32, 1, 1, 784), (25088, 1, 25088, 25088, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf59, buf60, buf61, buf62, 25088, 128, grid=grid(25088), stream=stream0)
        buf63 = empty_strided((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), device='cuda', dtype=torch.float32)
        buf64 = empty_strided((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), device='cuda', dtype=torch.float32)
        buf65 = empty_strided((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf60, buf61, buf62, buf63, buf64, buf65, 224, 112, grid=grid(224), stream=stream0)
        buf66 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf67 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf69 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf63, buf64, buf65, primals_324, primals_325, buf66, buf67, buf69, primals_324, primals_325, 32, 7, grid=grid(32), stream=stream0)
        del buf63
        del buf64
        del buf65
        del primals_324
        del primals_325
        buf70 = reinterpret_tensor(buf58, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf58  # reuse
        # Source Nodes: [l__mod___level1_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_17.run(buf59, buf66, buf67, primals_8, primals_9, buf70, 3211264, grid=grid(3211264), stream=stream0)
        del buf67
        del primals_9
        buf71 = empty_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda', dtype=torch.int64)
        # Source Nodes: [bottom], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_18.run(buf70, buf71, buf72, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [l__mod___level2_project_0], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf71, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf74 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level2_project_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf73, buf74, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        buf75 = reinterpret_tensor(buf62, (1, 128, 1, 1, 196), (25088, 1, 25088, 25088, 128), 0); del buf62  # reuse
        buf76 = reinterpret_tensor(buf61, (1, 128, 1, 1, 196), (25088, 1, 25088, 25088, 128), 0); del buf61  # reuse
        buf77 = reinterpret_tensor(buf60, (1, 128, 1, 1, 196), (25088, 1, 25088, 25088, 128), 0); del buf60  # reuse
        # Source Nodes: [shortcut], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf74, buf75, buf76, buf77, 25088, 128, grid=grid(25088), stream=stream0)
        buf78 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        buf79 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        buf80 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf75, buf76, buf77, buf78, buf79, buf80, 256, 98, grid=grid(256), stream=stream0)
        buf81 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf82 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf84 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf78, buf79, buf80, primals_327, primals_328, buf81, buf82, buf84, primals_327, primals_328, 128, 2, grid=grid(128), stream=stream0)
        del primals_327
        del primals_328
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf70, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf86 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf85, buf86, 512, 12544, grid=grid(512, 12544), stream=stream0)
        buf87 = reinterpret_tensor(buf49, (1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), 0); del buf49  # reuse
        buf88 = reinterpret_tensor(buf48, (1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), 0); del buf48  # reuse
        buf89 = reinterpret_tensor(buf47, (1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), 0); del buf47  # reuse
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf86, buf87, buf88, buf89, 50176, 128, grid=grid(50176), stream=stream0)
        buf90 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf92 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf87, buf88, buf89, buf90, buf91, buf92, 448, 112, grid=grid(448), stream=stream0)
        del buf87
        del buf88
        del buf89
        buf93 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf94 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf96 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf90, buf91, buf92, primals_330, primals_331, buf93, buf94, buf96, primals_330, primals_331, 64, 7, grid=grid(64), stream=stream0)
        del buf90
        del buf91
        del buf92
        del primals_330
        del primals_331
        buf97 = reinterpret_tensor(buf85, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf85  # reuse
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf86, buf93, buf94, primals_14, primals_15, buf97, 6422528, grid=grid(6422528), stream=stream0)
        del primals_15
        # Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf99 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf98, buf99, 512, 3136, grid=grid(512, 3136), stream=stream0)
        buf100 = empty_strided((1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), device='cuda', dtype=torch.float32)
        buf101 = empty_strided((1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), device='cuda', dtype=torch.float32)
        buf102 = empty_strided((1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf99, buf100, buf101, buf102, 12544, 128, grid=grid(12544), stream=stream0)
        buf103 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf104 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf105 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf100, buf101, buf102, buf103, buf104, buf105, 128, 98, grid=grid(128), stream=stream0)
        buf106 = buf94; del buf94  # reuse
        buf107 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf109 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf103, buf104, buf105, primals_333, primals_334, buf106, buf107, buf109, primals_333, primals_334, 64, 2, grid=grid(64), stream=stream0)
        del primals_333
        del primals_334
        buf110 = reinterpret_tensor(buf98, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf98  # reuse
        # Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_32.run(buf99, buf106, buf107, primals_17, primals_18, buf110, 1605632, grid=grid(1605632), stream=stream0)
        del primals_18
        # Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf112 = reinterpret_tensor(buf73, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf73  # reuse
        # Source Nodes: [out_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf111, buf112, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        buf113 = buf77; del buf77  # reuse
        buf114 = buf76; del buf76  # reuse
        buf115 = buf75; del buf75  # reuse
        # Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf112, buf113, buf114, buf115, 25088, 128, grid=grid(25088), stream=stream0)
        buf116 = buf80; del buf80  # reuse
        buf117 = buf79; del buf79  # reuse
        buf118 = buf78; del buf78  # reuse
        # Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf113, buf114, buf115, buf116, buf117, buf118, 256, 98, grid=grid(256), stream=stream0)
        buf119 = reinterpret_tensor(buf105, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf105  # reuse
        buf120 = reinterpret_tensor(buf104, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf104  # reuse
        buf122 = reinterpret_tensor(buf103, (128, ), (1, ), 0); del buf103  # reuse
        # Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf116, buf117, buf118, primals_336, primals_337, buf119, buf120, buf122, primals_336, primals_337, 128, 2, grid=grid(128), stream=stream0)
        del primals_336
        del primals_337
        buf123 = reinterpret_tensor(buf111, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf111  # reuse
        buf124 = buf123; del buf123  # reuse
        # Source Nodes: [out_7, out_8, shortcut, shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_33.run(buf124, buf112, buf119, buf120, primals_20, primals_21, buf74, buf81, buf82, primals_11, primals_12, 3211264, grid=grid(3211264), stream=stream0)
        del primals_12
        del primals_21
        # Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf126 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_10], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf125, buf126, 512, 3136, grid=grid(512, 3136), stream=stream0)
        buf127 = buf102; del buf102  # reuse
        buf128 = buf101; del buf101  # reuse
        buf129 = buf100; del buf100  # reuse
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf126, buf127, buf128, buf129, 12544, 128, grid=grid(12544), stream=stream0)
        buf130 = reinterpret_tensor(buf82, (1, 64, 1, 1, 2), (128, 1, 128, 128, 64), 0); del buf82  # reuse
        buf131 = reinterpret_tensor(buf120, (1, 64, 1, 1, 2), (128, 1, 128, 128, 64), 0); del buf120  # reuse
        buf132 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf127, buf128, buf129, buf130, buf131, buf132, 128, 98, grid=grid(128), stream=stream0)
        buf133 = buf107; del buf107  # reuse
        buf134 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf136 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf130, buf131, buf132, primals_339, primals_340, buf133, buf134, buf136, primals_339, primals_340, 64, 2, grid=grid(64), stream=stream0)
        del primals_339
        del primals_340
        buf137 = reinterpret_tensor(buf125, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf125  # reuse
        # Source Nodes: [out_11, out_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_32.run(buf126, buf133, buf134, primals_23, primals_24, buf137, 1605632, grid=grid(1605632), stream=stream0)
        del primals_24
        # Source Nodes: [out_13], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf139 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_13], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf138, buf139, 512, 3136, grid=grid(512, 3136), stream=stream0)
        buf140 = buf129; del buf129  # reuse
        buf141 = buf128; del buf128  # reuse
        buf142 = buf127; del buf127  # reuse
        # Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf139, buf140, buf141, buf142, 12544, 128, grid=grid(12544), stream=stream0)
        buf143 = buf132; del buf132  # reuse
        buf144 = buf131; del buf131  # reuse
        buf145 = buf130; del buf130  # reuse
        # Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf140, buf141, buf142, buf143, buf144, buf145, 128, 98, grid=grid(128), stream=stream0)
        buf146 = buf134; del buf134  # reuse
        buf147 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf149 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf143, buf144, buf145, primals_342, primals_343, buf146, buf147, buf149, primals_342, primals_343, 64, 2, grid=grid(64), stream=stream0)
        del primals_342
        del primals_343
        buf150 = reinterpret_tensor(buf138, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf138  # reuse
        # Source Nodes: [out_14, out_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_32.run(buf139, buf146, buf147, primals_26, primals_27, buf150, 1605632, grid=grid(1605632), stream=stream0)
        del buf147
        del primals_27
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf152 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf151, buf152, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        buf153 = buf115; del buf115  # reuse
        buf154 = buf114; del buf114  # reuse
        buf155 = buf113; del buf113  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf152, buf153, buf154, buf155, 25088, 128, grid=grid(25088), stream=stream0)
        buf156 = buf118; del buf118  # reuse
        buf157 = buf117; del buf117  # reuse
        buf158 = buf116; del buf116  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf153, buf154, buf155, buf156, buf157, buf158, 256, 98, grid=grid(256), stream=stream0)
        buf159 = reinterpret_tensor(buf145, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf145  # reuse
        buf160 = reinterpret_tensor(buf144, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf144  # reuse
        buf162 = reinterpret_tensor(buf143, (128, ), (1, ), 0); del buf143  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf156, buf157, buf158, primals_345, primals_346, buf159, buf160, buf162, primals_345, primals_346, 128, 2, grid=grid(128), stream=stream0)
        del primals_345
        del primals_346
        buf165 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        buf163 = reinterpret_tensor(buf165, (8, 128, 56, 56), (802816, 1, 14336, 256), 0)  # alias
        buf164 = reinterpret_tensor(buf165, (8, 128, 56, 56), (802816, 1, 14336, 256), 128)  # alias
        buf1185 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_27, out_17, out_18, x2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_34.run(buf152, buf159, buf160, primals_29, primals_30, buf124, buf163, buf164, buf1185, 3211264, grid=grid(3211264), stream=stream0)
        del primals_30
        # Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf167 = reinterpret_tensor(buf151, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf151  # reuse
        # Source Nodes: [x_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf166, buf167, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        buf168 = buf155; del buf155  # reuse
        buf169 = buf154; del buf154  # reuse
        buf170 = buf153; del buf153  # reuse
        # Source Nodes: [x_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf167, buf168, buf169, buf170, 25088, 128, grid=grid(25088), stream=stream0)
        buf171 = buf158; del buf158  # reuse
        buf172 = buf157; del buf157  # reuse
        buf173 = buf156; del buf156  # reuse
        # Source Nodes: [x_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf168, buf169, buf170, buf171, buf172, buf173, 256, 98, grid=grid(256), stream=stream0)
        buf174 = buf160; del buf160  # reuse
        buf175 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf177 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf171, buf172, buf173, primals_348, primals_349, buf174, buf175, buf177, primals_348, primals_349, 128, 2, grid=grid(128), stream=stream0)
        del primals_348
        del primals_349
        buf178 = reinterpret_tensor(buf166, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf166  # reuse
        # Source Nodes: [x_4, x_5, x_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_35.run(buf167, buf174, buf175, primals_32, primals_33, buf163, buf178, 3211264, grid=grid(3211264), stream=stream0)
        del primals_33
        buf179 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.int64)
        buf475 = empty_strided((8, 1152, 28, 28), (903168, 1, 32256, 1152), device='cuda', dtype=torch.float32)
        buf472 = reinterpret_tensor(buf475, (8, 128, 28, 28), (903168, 1, 32256, 1152), 512)  # alias
        # Source Nodes: [bottom_1, cat_23], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
        triton_poi_fused_cat_max_pool2d_with_indices_36.run(buf178, buf179, buf180, buf472, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [l__mod___level3_tree1_tree1_project_0], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf179, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf182 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level3_tree1_tree1_project_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf181, buf182, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf183 = reinterpret_tensor(buf142, (1, 256, 1, 1, 49), (12544, 1, 12544, 12544, 256), 0); del buf142  # reuse
        buf184 = reinterpret_tensor(buf141, (1, 256, 1, 1, 49), (12544, 1, 12544, 12544, 256), 0); del buf141  # reuse
        buf185 = reinterpret_tensor(buf140, (1, 256, 1, 1, 49), (12544, 1, 12544, 12544, 256), 0); del buf140  # reuse
        # Source Nodes: [shortcut_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf182, buf183, buf184, buf185, 12544, 128, grid=grid(12544), stream=stream0)
        buf186 = reinterpret_tensor(buf173, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf173  # reuse
        buf187 = reinterpret_tensor(buf172, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf172  # reuse
        buf189 = reinterpret_tensor(buf171, (256, ), (1, ), 0); del buf171  # reuse
        # Source Nodes: [shortcut_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf183, buf184, buf185, primals_351, primals_352, buf186, buf187, buf189, primals_351, primals_352, 256, 49, grid=grid(256), stream=stream0)
        del primals_351
        del primals_352
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf178, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf191 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf190, buf191, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        buf192 = buf170; del buf170  # reuse
        buf193 = buf169; del buf169  # reuse
        buf194 = buf168; del buf168  # reuse
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf191, buf192, buf193, buf194, 25088, 128, grid=grid(25088), stream=stream0)
        buf195 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        buf196 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        buf197 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf192, buf193, buf194, buf195, buf196, buf197, 256, 98, grid=grid(256), stream=stream0)
        del buf192
        del buf193
        del buf194
        buf198 = buf175; del buf175  # reuse
        buf199 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf201 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf195, buf196, buf197, primals_354, primals_355, buf198, buf199, buf201, primals_354, primals_355, 128, 2, grid=grid(128), stream=stream0)
        del primals_354
        del primals_355
        buf202 = reinterpret_tensor(buf190, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf190  # reuse
        # Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_40.run(buf191, buf198, buf199, primals_38, primals_39, buf202, 3211264, grid=grid(3211264), stream=stream0)
        del primals_39
        # Source Nodes: [out_23], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf204 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf203, buf204, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf205 = empty_strided((1, 128, 1, 1, 49), (6272, 1, 6272, 6272, 128), device='cuda', dtype=torch.float32)
        buf206 = empty_strided((1, 128, 1, 1, 49), (6272, 1, 6272, 6272, 128), device='cuda', dtype=torch.float32)
        buf207 = empty_strided((1, 128, 1, 1, 49), (6272, 1, 6272, 6272, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf204, buf205, buf206, buf207, 6272, 128, grid=grid(6272), stream=stream0)
        buf208 = buf199; del buf199  # reuse
        buf209 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf211 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf205, buf206, buf207, primals_357, primals_358, buf208, buf209, buf211, primals_357, primals_358, 128, 49, grid=grid(128), stream=stream0)
        del primals_357
        del primals_358
        buf212 = reinterpret_tensor(buf203, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf203  # reuse
        # Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf204, buf208, buf209, primals_41, primals_42, buf212, 802816, grid=grid(802816), stream=stream0)
        del primals_42
        # Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf214 = reinterpret_tensor(buf181, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf181  # reuse
        # Source Nodes: [out_26], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf213, buf214, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf215 = buf185; del buf185  # reuse
        buf216 = buf184; del buf184  # reuse
        buf217 = buf183; del buf183  # reuse
        # Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf214, buf215, buf216, buf217, 12544, 128, grid=grid(12544), stream=stream0)
        buf218 = reinterpret_tensor(buf197, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf197  # reuse
        buf219 = reinterpret_tensor(buf196, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf196  # reuse
        buf221 = reinterpret_tensor(buf195, (256, ), (1, ), 0); del buf195  # reuse
        # Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf215, buf216, buf217, primals_360, primals_361, buf218, buf219, buf221, primals_360, primals_361, 256, 49, grid=grid(256), stream=stream0)
        del primals_360
        del primals_361
        buf222 = reinterpret_tensor(buf213, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf213  # reuse
        buf223 = buf222; del buf222  # reuse
        # Source Nodes: [out_27, out_28, shortcut_4, shortcut_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_45.run(buf223, buf214, buf218, buf219, primals_44, primals_45, buf182, buf186, buf187, primals_35, primals_36, 1605632, grid=grid(1605632), stream=stream0)
        del primals_36
        del primals_45
        # Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf225 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf224, buf225, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf226 = buf207; del buf207  # reuse
        buf227 = buf206; del buf206  # reuse
        buf228 = buf205; del buf205  # reuse
        # Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf225, buf226, buf227, buf228, 6272, 128, grid=grid(6272), stream=stream0)
        buf229 = buf209; del buf209  # reuse
        buf230 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf232 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf226, buf227, buf228, primals_363, primals_364, buf229, buf230, buf232, primals_363, primals_364, 128, 49, grid=grid(128), stream=stream0)
        del primals_363
        del primals_364
        buf233 = reinterpret_tensor(buf224, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf224  # reuse
        # Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf225, buf229, buf230, primals_47, primals_48, buf233, 802816, grid=grid(802816), stream=stream0)
        del primals_48
        # Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf235 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf234, buf235, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf236 = buf228; del buf228  # reuse
        buf237 = buf227; del buf227  # reuse
        buf238 = buf226; del buf226  # reuse
        # Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf235, buf236, buf237, buf238, 6272, 128, grid=grid(6272), stream=stream0)
        buf239 = buf230; del buf230  # reuse
        buf240 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf242 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf236, buf237, buf238, primals_366, primals_367, buf239, buf240, buf242, primals_366, primals_367, 128, 49, grid=grid(128), stream=stream0)
        del primals_366
        del primals_367
        buf243 = reinterpret_tensor(buf234, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf234  # reuse
        # Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf235, buf239, buf240, primals_50, primals_51, buf243, 802816, grid=grid(802816), stream=stream0)
        del primals_51
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf245 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf244, buf245, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf246 = buf217; del buf217  # reuse
        buf247 = buf216; del buf216  # reuse
        buf248 = buf215; del buf215  # reuse
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf245, buf246, buf247, buf248, 12544, 128, grid=grid(12544), stream=stream0)
        buf249 = buf219; del buf219  # reuse
        buf250 = buf187; del buf187  # reuse
        buf252 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf246, buf247, buf248, primals_369, primals_370, buf249, buf250, buf252, primals_369, primals_370, 256, 49, grid=grid(256), stream=stream0)
        del primals_369
        del primals_370
        buf255 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        buf253 = reinterpret_tensor(buf255, (8, 256, 28, 28), (401408, 1, 14336, 512), 0)  # alias
        buf254 = reinterpret_tensor(buf255, (8, 256, 28, 28), (401408, 1, 14336, 512), 256)  # alias
        buf1184 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_26, out_37, out_38, x2_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_46.run(buf245, buf249, buf250, primals_53, primals_54, buf223, buf253, buf254, buf1184, 1605632, grid=grid(1605632), stream=stream0)
        del primals_54
        # Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf257 = reinterpret_tensor(buf244, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf244  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf256, buf257, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf258 = buf248; del buf248  # reuse
        buf259 = buf247; del buf247  # reuse
        buf260 = buf246; del buf246  # reuse
        # Source Nodes: [x_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf257, buf258, buf259, buf260, 12544, 128, grid=grid(12544), stream=stream0)
        buf261 = buf250; del buf250  # reuse
        buf262 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf264 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf258, buf259, buf260, primals_372, primals_373, buf261, buf262, buf264, primals_372, primals_373, 256, 49, grid=grid(256), stream=stream0)
        del primals_372
        del primals_373
        buf265 = reinterpret_tensor(buf256, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf256  # reuse
        # Source Nodes: [x1_2, x_10, x_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_47.run(buf257, buf261, buf262, primals_56, primals_57, buf253, buf265, 1605632, grid=grid(1605632), stream=stream0)
        del primals_57
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf267 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf266, buf267, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf268 = buf238; del buf238  # reuse
        buf269 = buf237; del buf237  # reuse
        buf270 = buf236; del buf236  # reuse
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf267, buf268, buf269, buf270, 6272, 128, grid=grid(6272), stream=stream0)
        buf271 = buf240; del buf240  # reuse
        buf272 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf274 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf268, buf269, buf270, primals_375, primals_376, buf271, buf272, buf274, primals_375, primals_376, 128, 49, grid=grid(128), stream=stream0)
        del primals_375
        del primals_376
        buf275 = reinterpret_tensor(buf266, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf266  # reuse
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf267, buf271, buf272, primals_59, primals_60, buf275, 802816, grid=grid(802816), stream=stream0)
        del primals_60
        # Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf277 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf276, buf277, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf278 = buf270; del buf270  # reuse
        buf279 = buf269; del buf269  # reuse
        buf280 = buf268; del buf268  # reuse
        # Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf277, buf278, buf279, buf280, 6272, 128, grid=grid(6272), stream=stream0)
        buf281 = buf272; del buf272  # reuse
        buf282 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf284 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf278, buf279, buf280, primals_378, primals_379, buf281, buf282, buf284, primals_378, primals_379, 128, 49, grid=grid(128), stream=stream0)
        del primals_378
        del primals_379
        buf285 = reinterpret_tensor(buf276, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf276  # reuse
        # Source Nodes: [out_44, out_45], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf277, buf281, buf282, primals_62, primals_63, buf285, 802816, grid=grid(802816), stream=stream0)
        del primals_63
        # Source Nodes: [out_46], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf287 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_46], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf286, buf287, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf288 = buf260; del buf260  # reuse
        buf289 = buf259; del buf259  # reuse
        buf290 = buf258; del buf258  # reuse
        # Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf287, buf288, buf289, buf290, 12544, 128, grid=grid(12544), stream=stream0)
        buf291 = buf262; del buf262  # reuse
        buf292 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf294 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf288, buf289, buf290, primals_381, primals_382, buf291, buf292, buf294, primals_381, primals_382, 256, 49, grid=grid(256), stream=stream0)
        del primals_381
        del primals_382
        buf295 = reinterpret_tensor(buf286, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf286  # reuse
        buf328 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf327 = reinterpret_tensor(buf328, (8, 256, 28, 28), (602112, 1, 21504, 768), 512)  # alias
        # Source Nodes: [cat_25, out_47, out_48, shortcut_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_48.run(buf287, buf291, buf292, primals_65, primals_66, buf265, buf295, buf327, 1605632, grid=grid(1605632), stream=stream0)
        del primals_66
        # Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf297 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_50], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf296, buf297, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf298 = buf280; del buf280  # reuse
        buf299 = buf279; del buf279  # reuse
        buf300 = buf278; del buf278  # reuse
        # Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf297, buf298, buf299, buf300, 6272, 128, grid=grid(6272), stream=stream0)
        buf301 = buf282; del buf282  # reuse
        buf302 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf304 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf298, buf299, buf300, primals_384, primals_385, buf301, buf302, buf304, primals_384, primals_385, 128, 49, grid=grid(128), stream=stream0)
        del primals_384
        del primals_385
        buf305 = reinterpret_tensor(buf296, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf296  # reuse
        # Source Nodes: [out_51, out_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf297, buf301, buf302, primals_68, primals_69, buf305, 802816, grid=grid(802816), stream=stream0)
        del primals_69
        # Source Nodes: [out_53], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf307 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_53], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf306, buf307, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf308 = buf300; del buf300  # reuse
        buf309 = buf299; del buf299  # reuse
        buf310 = buf298; del buf298  # reuse
        # Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf307, buf308, buf309, buf310, 6272, 128, grid=grid(6272), stream=stream0)
        buf311 = buf302; del buf302  # reuse
        buf312 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf314 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf308, buf309, buf310, primals_387, primals_388, buf311, buf312, buf314, primals_387, primals_388, 128, 49, grid=grid(128), stream=stream0)
        del primals_387
        del primals_388
        buf315 = reinterpret_tensor(buf306, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf306  # reuse
        # Source Nodes: [out_54, out_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf307, buf311, buf312, primals_71, primals_72, buf315, 802816, grid=grid(802816), stream=stream0)
        del primals_72
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf317 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf316, buf317, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf318 = buf290; del buf290  # reuse
        buf319 = buf289; del buf289  # reuse
        buf320 = buf288; del buf288  # reuse
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf317, buf318, buf319, buf320, 12544, 128, grid=grid(12544), stream=stream0)
        buf321 = buf292; del buf292  # reuse
        buf322 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf324 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf318, buf319, buf320, primals_390, primals_391, buf321, buf322, buf324, primals_390, primals_391, 256, 49, grid=grid(256), stream=stream0)
        del primals_390
        del primals_391
        buf325 = reinterpret_tensor(buf328, (8, 256, 28, 28), (602112, 1, 21504, 768), 0)  # alias
        buf326 = reinterpret_tensor(buf328, (8, 256, 28, 28), (602112, 1, 21504, 768), 256)  # alias
        buf1183 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_25, out_57, out_58, x2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_49.run(buf317, buf321, buf322, primals_74, primals_75, buf295, buf325, buf326, buf1183, 1605632, grid=grid(1605632), stream=stream0)
        del primals_75
        # Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf330 = reinterpret_tensor(buf316, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf316  # reuse
        # Source Nodes: [x_14], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf329, buf330, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf331 = buf320; del buf320  # reuse
        buf332 = buf319; del buf319  # reuse
        buf333 = buf318; del buf318  # reuse
        # Source Nodes: [x_15], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf330, buf331, buf332, buf333, 12544, 128, grid=grid(12544), stream=stream0)
        buf334 = buf322; del buf322  # reuse
        buf335 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf337 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_15], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf331, buf332, buf333, primals_393, primals_394, buf334, buf335, buf337, primals_393, primals_394, 256, 49, grid=grid(256), stream=stream0)
        del primals_393
        del primals_394
        buf338 = reinterpret_tensor(buf329, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf329  # reuse
        buf473 = reinterpret_tensor(buf475, (8, 256, 28, 28), (903168, 1, 32256, 1152), 640)  # alias
        # Source Nodes: [cat_23, x1_4, x_15, x_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_50.run(buf330, buf334, buf335, primals_77, primals_78, buf325, buf338, buf473, 1605632, grid=grid(1605632), stream=stream0)
        del primals_78
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf340 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf339, buf340, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf341 = buf310; del buf310  # reuse
        buf342 = buf309; del buf309  # reuse
        buf343 = buf308; del buf308  # reuse
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf340, buf341, buf342, buf343, 6272, 128, grid=grid(6272), stream=stream0)
        buf344 = buf312; del buf312  # reuse
        buf345 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf347 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf341, buf342, buf343, primals_396, primals_397, buf344, buf345, buf347, primals_396, primals_397, 128, 49, grid=grid(128), stream=stream0)
        del primals_396
        del primals_397
        buf348 = reinterpret_tensor(buf339, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf339  # reuse
        # Source Nodes: [out_61, out_62], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf340, buf344, buf345, primals_80, primals_81, buf348, 802816, grid=grid(802816), stream=stream0)
        del primals_81
        # Source Nodes: [out_63], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf348, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf350 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf349, buf350, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf351 = buf343; del buf343  # reuse
        buf352 = buf342; del buf342  # reuse
        buf353 = buf341; del buf341  # reuse
        # Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf350, buf351, buf352, buf353, 6272, 128, grid=grid(6272), stream=stream0)
        buf354 = buf345; del buf345  # reuse
        buf355 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf357 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf351, buf352, buf353, primals_399, primals_400, buf354, buf355, buf357, primals_399, primals_400, 128, 49, grid=grid(128), stream=stream0)
        del primals_399
        del primals_400
        buf358 = reinterpret_tensor(buf349, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf349  # reuse
        # Source Nodes: [out_64, out_65], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf350, buf354, buf355, primals_83, primals_84, buf358, 802816, grid=grid(802816), stream=stream0)
        del primals_84
        # Source Nodes: [out_66], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf360 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf359, buf360, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf361 = buf333; del buf333  # reuse
        buf362 = buf332; del buf332  # reuse
        buf363 = buf331; del buf331  # reuse
        # Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf360, buf361, buf362, buf363, 12544, 128, grid=grid(12544), stream=stream0)
        buf364 = buf335; del buf335  # reuse
        buf365 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf367 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf361, buf362, buf363, primals_402, primals_403, buf364, buf365, buf367, primals_402, primals_403, 256, 49, grid=grid(256), stream=stream0)
        del primals_402
        del primals_403
        buf368 = reinterpret_tensor(buf359, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf359  # reuse
        # Source Nodes: [out_67, out_68, shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_51.run(buf360, buf364, buf365, primals_86, primals_87, buf338, buf368, 1605632, grid=grid(1605632), stream=stream0)
        del primals_87
        # Source Nodes: [out_70], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf370 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_70], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf369, buf370, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf371 = buf353; del buf353  # reuse
        buf372 = buf352; del buf352  # reuse
        buf373 = buf351; del buf351  # reuse
        # Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf370, buf371, buf372, buf373, 6272, 128, grid=grid(6272), stream=stream0)
        buf374 = buf355; del buf355  # reuse
        buf375 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf377 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf371, buf372, buf373, primals_405, primals_406, buf374, buf375, buf377, primals_405, primals_406, 128, 49, grid=grid(128), stream=stream0)
        del primals_405
        del primals_406
        buf378 = reinterpret_tensor(buf369, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf369  # reuse
        # Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf370, buf374, buf375, primals_89, primals_90, buf378, 802816, grid=grid(802816), stream=stream0)
        del primals_90
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf380 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf379, buf380, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf381 = buf373; del buf373  # reuse
        buf382 = buf372; del buf372  # reuse
        buf383 = buf371; del buf371  # reuse
        # Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf380, buf381, buf382, buf383, 6272, 128, grid=grid(6272), stream=stream0)
        buf384 = buf375; del buf375  # reuse
        buf385 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf387 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf381, buf382, buf383, primals_408, primals_409, buf384, buf385, buf387, primals_408, primals_409, 128, 49, grid=grid(128), stream=stream0)
        del primals_408
        del primals_409
        buf388 = reinterpret_tensor(buf379, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf379  # reuse
        # Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf380, buf384, buf385, primals_92, primals_93, buf388, 802816, grid=grid(802816), stream=stream0)
        del primals_93
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf390 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf389, buf390, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf391 = buf363; del buf363  # reuse
        buf392 = buf362; del buf362  # reuse
        buf393 = buf361; del buf361  # reuse
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf390, buf391, buf392, buf393, 12544, 128, grid=grid(12544), stream=stream0)
        buf394 = buf365; del buf365  # reuse
        buf395 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf397 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf391, buf392, buf393, primals_411, primals_412, buf394, buf395, buf397, primals_411, primals_412, 256, 49, grid=grid(256), stream=stream0)
        del primals_411
        del primals_412
        buf400 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        buf398 = reinterpret_tensor(buf400, (8, 256, 28, 28), (401408, 1, 14336, 512), 0)  # alias
        buf399 = reinterpret_tensor(buf400, (8, 256, 28, 28), (401408, 1, 14336, 512), 256)  # alias
        buf1182 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_24, out_77, out_78, x2_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_46.run(buf390, buf394, buf395, primals_95, primals_96, buf368, buf398, buf399, buf1182, 1605632, grid=grid(1605632), stream=stream0)
        del primals_96
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf402 = reinterpret_tensor(buf389, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf389  # reuse
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf401, buf402, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf403 = buf393; del buf393  # reuse
        buf404 = buf392; del buf392  # reuse
        buf405 = buf391; del buf391  # reuse
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf402, buf403, buf404, buf405, 12544, 128, grid=grid(12544), stream=stream0)
        buf406 = buf395; del buf395  # reuse
        buf407 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf409 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf403, buf404, buf405, primals_414, primals_415, buf406, buf407, buf409, primals_414, primals_415, 256, 49, grid=grid(256), stream=stream0)
        del primals_414
        del primals_415
        buf410 = reinterpret_tensor(buf401, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf401  # reuse
        # Source Nodes: [x1_6, x_21, x_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_47.run(buf402, buf406, buf407, primals_98, primals_99, buf398, buf410, 1605632, grid=grid(1605632), stream=stream0)
        del primals_99
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf410, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf412 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf411, buf412, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf413 = buf383; del buf383  # reuse
        buf414 = buf382; del buf382  # reuse
        buf415 = buf381; del buf381  # reuse
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf412, buf413, buf414, buf415, 6272, 128, grid=grid(6272), stream=stream0)
        buf416 = buf385; del buf385  # reuse
        buf417 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf419 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf413, buf414, buf415, primals_417, primals_418, buf416, buf417, buf419, primals_417, primals_418, 128, 49, grid=grid(128), stream=stream0)
        del primals_417
        del primals_418
        buf420 = reinterpret_tensor(buf411, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf411  # reuse
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf412, buf416, buf417, primals_101, primals_102, buf420, 802816, grid=grid(802816), stream=stream0)
        del primals_102
        # Source Nodes: [out_83], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf420, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf422 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_83], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf421, buf422, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf423 = buf415; del buf415  # reuse
        buf424 = buf414; del buf414  # reuse
        buf425 = buf413; del buf413  # reuse
        # Source Nodes: [out_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf422, buf423, buf424, buf425, 6272, 128, grid=grid(6272), stream=stream0)
        buf426 = buf417; del buf417  # reuse
        buf427 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf429 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf423, buf424, buf425, primals_420, primals_421, buf426, buf427, buf429, primals_420, primals_421, 128, 49, grid=grid(128), stream=stream0)
        del primals_420
        del primals_421
        buf430 = reinterpret_tensor(buf421, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf421  # reuse
        # Source Nodes: [out_84, out_85], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf422, buf426, buf427, primals_104, primals_105, buf430, 802816, grid=grid(802816), stream=stream0)
        del primals_105
        # Source Nodes: [out_86], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf432 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_86], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf431, buf432, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf433 = buf405; del buf405  # reuse
        buf434 = buf404; del buf404  # reuse
        buf435 = buf403; del buf403  # reuse
        # Source Nodes: [out_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf432, buf433, buf434, buf435, 12544, 128, grid=grid(12544), stream=stream0)
        buf436 = buf407; del buf407  # reuse
        buf437 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf439 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf433, buf434, buf435, primals_423, primals_424, buf436, buf437, buf439, primals_423, primals_424, 256, 49, grid=grid(256), stream=stream0)
        del primals_423
        del primals_424
        buf440 = reinterpret_tensor(buf431, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf431  # reuse
        buf474 = reinterpret_tensor(buf475, (8, 256, 28, 28), (903168, 1, 32256, 1152), 896)  # alias
        # Source Nodes: [cat_23, out_87, out_88, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_52.run(buf432, buf436, buf437, primals_107, primals_108, buf410, buf440, buf474, 1605632, grid=grid(1605632), stream=stream0)
        del primals_108
        # Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf441 = extern_kernels.convolution(buf440, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf441, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf442 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_90], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf441, buf442, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf443 = buf425; del buf425  # reuse
        buf444 = buf424; del buf424  # reuse
        buf445 = buf423; del buf423  # reuse
        # Source Nodes: [out_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf442, buf443, buf444, buf445, 6272, 128, grid=grid(6272), stream=stream0)
        buf446 = buf427; del buf427  # reuse
        buf447 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf449 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf443, buf444, buf445, primals_426, primals_427, buf446, buf447, buf449, primals_426, primals_427, 128, 49, grid=grid(128), stream=stream0)
        del primals_426
        del primals_427
        buf450 = reinterpret_tensor(buf441, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf441  # reuse
        # Source Nodes: [out_91, out_92], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf442, buf446, buf447, primals_110, primals_111, buf450, 802816, grid=grid(802816), stream=stream0)
        del primals_111
        # Source Nodes: [out_93], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf450, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf452 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_93], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf451, buf452, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf453 = buf445; del buf445  # reuse
        buf454 = buf444; del buf444  # reuse
        buf455 = buf443; del buf443  # reuse
        # Source Nodes: [out_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf452, buf453, buf454, buf455, 6272, 128, grid=grid(6272), stream=stream0)
        buf456 = buf447; del buf447  # reuse
        buf457 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf459 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf453, buf454, buf455, primals_429, primals_430, buf456, buf457, buf459, primals_429, primals_430, 128, 49, grid=grid(128), stream=stream0)
        del buf453
        del buf454
        del buf455
        del primals_429
        del primals_430
        buf460 = reinterpret_tensor(buf451, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf451  # reuse
        # Source Nodes: [out_94, out_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf452, buf456, buf457, primals_113, primals_114, buf460, 802816, grid=grid(802816), stream=stream0)
        del buf457
        del primals_114
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf461 = extern_kernels.convolution(buf460, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf461, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf462 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf461, buf462, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf463 = buf435; del buf435  # reuse
        buf464 = buf434; del buf434  # reuse
        buf465 = buf433; del buf433  # reuse
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf462, buf463, buf464, buf465, 12544, 128, grid=grid(12544), stream=stream0)
        buf466 = buf437; del buf437  # reuse
        buf467 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf469 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf463, buf464, buf465, primals_432, primals_433, buf466, buf467, buf469, primals_432, primals_433, 256, 49, grid=grid(256), stream=stream0)
        del primals_432
        del primals_433
        buf470 = reinterpret_tensor(buf475, (8, 256, 28, 28), (903168, 1, 32256, 1152), 0)  # alias
        buf471 = reinterpret_tensor(buf475, (8, 256, 28, 28), (903168, 1, 32256, 1152), 256)  # alias
        buf1181 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_23, out_97, out_98, x2_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_53.run(buf462, buf466, buf467, primals_116, primals_117, buf440, buf470, buf471, buf1181, 1605632, grid=grid(1605632), stream=stream0)
        del primals_117
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf475, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf477 = reinterpret_tensor(buf461, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf461  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf476, buf477, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf478 = buf465; del buf465  # reuse
        buf479 = buf464; del buf464  # reuse
        buf480 = buf463; del buf463  # reuse
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf477, buf478, buf479, buf480, 12544, 128, grid=grid(12544), stream=stream0)
        buf481 = buf467; del buf467  # reuse
        buf482 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf484 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf478, buf479, buf480, primals_435, primals_436, buf481, buf482, buf484, primals_435, primals_436, 256, 49, grid=grid(256), stream=stream0)
        del primals_435
        del primals_436
        buf485 = reinterpret_tensor(buf476, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf476  # reuse
        # Source Nodes: [x_26, x_27, x_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_54.run(buf477, buf481, buf482, primals_119, primals_120, buf470, buf485, 1605632, grid=grid(1605632), stream=stream0)
        del primals_120
        buf486 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        buf487 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.int64)
        buf1071 = empty_strided((8, 2816, 14, 14), (551936, 1, 39424, 2816), device='cuda', dtype=torch.float32)
        buf1067 = reinterpret_tensor(buf1071, (8, 256, 14, 14), (551936, 1, 39424, 2816), 1024)  # alias
        # Source Nodes: [bottom_8, cat_15], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
        triton_poi_fused_cat_max_pool2d_with_indices_55.run(buf485, buf486, buf487, buf1067, 401408, grid=grid(401408), stream=stream0)
        # Source Nodes: [l__mod___level4_tree1_tree1_tree1_project_0], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(buf486, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf489 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level4_tree1_tree1_tree1_project_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf488, buf489, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf490 = empty_strided((1, 512, 1, 1, 13), (6656, 1, 6656, 6656, 512), device='cuda', dtype=torch.float32)
        buf491 = empty_strided((1, 512, 1, 1, 13), (6656, 1, 6656, 6656, 512), device='cuda', dtype=torch.float32)
        buf492 = empty_strided((1, 512, 1, 1, 13), (6656, 1, 6656, 6656, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_16], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf489, buf490, buf491, buf492, 6656, 121, grid=grid(6656), stream=stream0)
        buf493 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf494 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf496 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_16], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf490, buf491, buf492, primals_438, primals_439, buf493, buf494, buf496, primals_438, primals_439, 512, 13, grid=grid(512), stream=stream0)
        del primals_438
        del primals_439
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf485, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf498 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf497, buf498, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf499 = buf480; del buf480  # reuse
        buf500 = buf479; del buf479  # reuse
        buf501 = buf478; del buf478  # reuse
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf498, buf499, buf500, buf501, 12544, 128, grid=grid(12544), stream=stream0)
        buf502 = buf482; del buf482  # reuse
        buf503 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf505 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf499, buf500, buf501, primals_441, primals_442, buf502, buf503, buf505, primals_441, primals_442, 256, 49, grid=grid(256), stream=stream0)
        del buf499
        del buf500
        del buf501
        del primals_441
        del primals_442
        buf506 = reinterpret_tensor(buf497, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf497  # reuse
        # Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_59.run(buf498, buf502, buf503, primals_125, primals_126, buf506, 1605632, grid=grid(1605632), stream=stream0)
        del primals_126
        # Source Nodes: [out_103], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf506, buf13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf508 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_103], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf507, buf508, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf509 = empty_strided((1, 256, 1, 1, 13), (3328, 1, 3328, 3328, 256), device='cuda', dtype=torch.float32)
        buf510 = empty_strided((1, 256, 1, 1, 13), (3328, 1, 3328, 3328, 256), device='cuda', dtype=torch.float32)
        buf511 = empty_strided((1, 256, 1, 1, 13), (3328, 1, 3328, 3328, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf508, buf509, buf510, buf511, 3328, 121, grid=grid(3328), stream=stream0)
        buf512 = buf503; del buf503  # reuse
        buf513 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf515 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf509, buf510, buf511, primals_444, primals_445, buf512, buf513, buf515, primals_444, primals_445, 256, 13, grid=grid(256), stream=stream0)
        del primals_444
        del primals_445
        buf516 = reinterpret_tensor(buf507, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf507  # reuse
        # Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf508, buf512, buf513, primals_128, primals_129, buf516, 401408, grid=grid(401408), stream=stream0)
        del primals_129
        # Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf518 = reinterpret_tensor(buf488, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf488  # reuse
        # Source Nodes: [out_106], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf517, buf518, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf519 = buf492; del buf492  # reuse
        buf520 = buf491; del buf491  # reuse
        buf521 = buf490; del buf490  # reuse
        # Source Nodes: [out_107], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf518, buf519, buf520, buf521, 6656, 121, grid=grid(6656), stream=stream0)
        buf522 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf523 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf525 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_107], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf519, buf520, buf521, primals_447, primals_448, buf522, buf523, buf525, primals_447, primals_448, 512, 13, grid=grid(512), stream=stream0)
        del primals_447
        del primals_448
        buf526 = reinterpret_tensor(buf517, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf517  # reuse
        buf527 = buf526; del buf526  # reuse
        # Source Nodes: [out_107, out_108, shortcut_16, shortcut_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_64.run(buf527, buf518, buf522, buf523, primals_131, primals_132, buf489, buf493, buf494, primals_122, primals_123, 802816, grid=grid(802816), stream=stream0)
        del primals_123
        del primals_132
        # Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf528 = extern_kernels.convolution(buf527, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf528, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf529 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_110], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf528, buf529, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf530 = buf511; del buf511  # reuse
        buf531 = buf510; del buf510  # reuse
        buf532 = buf509; del buf509  # reuse
        # Source Nodes: [out_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf529, buf530, buf531, buf532, 3328, 121, grid=grid(3328), stream=stream0)
        buf533 = buf513; del buf513  # reuse
        buf534 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf536 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf530, buf531, buf532, primals_450, primals_451, buf533, buf534, buf536, primals_450, primals_451, 256, 13, grid=grid(256), stream=stream0)
        del primals_450
        del primals_451
        buf537 = reinterpret_tensor(buf528, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf528  # reuse
        # Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf529, buf533, buf534, primals_134, primals_135, buf537, 401408, grid=grid(401408), stream=stream0)
        del primals_135
        # Source Nodes: [out_113], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf539 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_113], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf538, buf539, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf540 = buf532; del buf532  # reuse
        buf541 = buf531; del buf531  # reuse
        buf542 = buf530; del buf530  # reuse
        # Source Nodes: [out_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf539, buf540, buf541, buf542, 3328, 121, grid=grid(3328), stream=stream0)
        buf543 = buf534; del buf534  # reuse
        buf544 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf546 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf540, buf541, buf542, primals_453, primals_454, buf543, buf544, buf546, primals_453, primals_454, 256, 13, grid=grid(256), stream=stream0)
        del primals_453
        del primals_454
        buf547 = reinterpret_tensor(buf538, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf538  # reuse
        # Source Nodes: [out_114, out_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf539, buf543, buf544, primals_137, primals_138, buf547, 401408, grid=grid(401408), stream=stream0)
        del primals_138
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(buf547, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf549 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf548, buf549, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf550 = buf521; del buf521  # reuse
        buf551 = buf520; del buf520  # reuse
        buf552 = buf519; del buf519  # reuse
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf549, buf550, buf551, buf552, 6656, 121, grid=grid(6656), stream=stream0)
        buf553 = buf523; del buf523  # reuse
        buf554 = buf494; del buf494  # reuse
        buf556 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf550, buf551, buf552, primals_456, primals_457, buf553, buf554, buf556, primals_456, primals_457, 512, 13, grid=grid(512), stream=stream0)
        del primals_456
        del primals_457
        buf559 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        buf557 = reinterpret_tensor(buf559, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf558 = reinterpret_tensor(buf559, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        buf1180 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_22, out_117, out_118, x2_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_65.run(buf549, buf553, buf554, primals_140, primals_141, buf527, buf557, buf558, buf1180, 802816, grid=grid(802816), stream=stream0)
        del primals_141
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf559, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf561 = reinterpret_tensor(buf548, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf548  # reuse
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf560, buf561, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf562 = buf552; del buf552  # reuse
        buf563 = buf551; del buf551  # reuse
        buf564 = buf550; del buf550  # reuse
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf561, buf562, buf563, buf564, 6656, 121, grid=grid(6656), stream=stream0)
        buf565 = buf554; del buf554  # reuse
        buf566 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf568 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf562, buf563, buf564, primals_459, primals_460, buf565, buf566, buf568, primals_459, primals_460, 512, 13, grid=grid(512), stream=stream0)
        del primals_459
        del primals_460
        buf569 = reinterpret_tensor(buf560, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf560  # reuse
        # Source Nodes: [x1_9, x_34, x_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_66.run(buf561, buf565, buf566, primals_143, primals_144, buf557, buf569, 802816, grid=grid(802816), stream=stream0)
        del primals_144
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf571 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf570, buf571, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf572 = buf542; del buf542  # reuse
        buf573 = buf541; del buf541  # reuse
        buf574 = buf540; del buf540  # reuse
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf571, buf572, buf573, buf574, 3328, 121, grid=grid(3328), stream=stream0)
        buf575 = buf544; del buf544  # reuse
        buf576 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf578 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf572, buf573, buf574, primals_462, primals_463, buf575, buf576, buf578, primals_462, primals_463, 256, 13, grid=grid(256), stream=stream0)
        del primals_462
        del primals_463
        buf579 = reinterpret_tensor(buf570, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf570  # reuse
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf571, buf575, buf576, primals_146, primals_147, buf579, 401408, grid=grid(401408), stream=stream0)
        del primals_147
        # Source Nodes: [out_123], Original ATen: [aten.convolution]
        buf580 = extern_kernels.convolution(buf579, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf580, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf581 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_123], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf580, buf581, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf582 = buf574; del buf574  # reuse
        buf583 = buf573; del buf573  # reuse
        buf584 = buf572; del buf572  # reuse
        # Source Nodes: [out_124], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf581, buf582, buf583, buf584, 3328, 121, grid=grid(3328), stream=stream0)
        buf585 = buf576; del buf576  # reuse
        buf586 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf588 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_124], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf582, buf583, buf584, primals_465, primals_466, buf585, buf586, buf588, primals_465, primals_466, 256, 13, grid=grid(256), stream=stream0)
        del primals_465
        del primals_466
        buf589 = reinterpret_tensor(buf580, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf580  # reuse
        # Source Nodes: [out_124, out_125], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf581, buf585, buf586, primals_149, primals_150, buf589, 401408, grid=grid(401408), stream=stream0)
        del primals_150
        # Source Nodes: [out_126], Original ATen: [aten.convolution]
        buf590 = extern_kernels.convolution(buf589, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf590, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf591 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_126], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf590, buf591, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf592 = buf564; del buf564  # reuse
        buf593 = buf563; del buf563  # reuse
        buf594 = buf562; del buf562  # reuse
        # Source Nodes: [out_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf591, buf592, buf593, buf594, 6656, 121, grid=grid(6656), stream=stream0)
        buf595 = buf566; del buf566  # reuse
        buf596 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf598 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf592, buf593, buf594, primals_468, primals_469, buf595, buf596, buf598, primals_468, primals_469, 512, 13, grid=grid(512), stream=stream0)
        del primals_468
        del primals_469
        buf599 = reinterpret_tensor(buf590, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf590  # reuse
        buf632 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        buf631 = reinterpret_tensor(buf632, (8, 512, 14, 14), (301056, 1, 21504, 1536), 1024)  # alias
        # Source Nodes: [cat_21, out_127, out_128, shortcut_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_67.run(buf591, buf595, buf596, primals_152, primals_153, buf569, buf599, buf631, 802816, grid=grid(802816), stream=stream0)
        del primals_153
        # Source Nodes: [out_130], Original ATen: [aten.convolution]
        buf600 = extern_kernels.convolution(buf599, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf601 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_130], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf600, buf601, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf602 = buf584; del buf584  # reuse
        buf603 = buf583; del buf583  # reuse
        buf604 = buf582; del buf582  # reuse
        # Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf601, buf602, buf603, buf604, 3328, 121, grid=grid(3328), stream=stream0)
        buf605 = buf586; del buf586  # reuse
        buf606 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf608 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf602, buf603, buf604, primals_471, primals_472, buf605, buf606, buf608, primals_471, primals_472, 256, 13, grid=grid(256), stream=stream0)
        del primals_471
        del primals_472
        buf609 = reinterpret_tensor(buf600, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf600  # reuse
        # Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf601, buf605, buf606, primals_155, primals_156, buf609, 401408, grid=grid(401408), stream=stream0)
        del primals_156
        # Source Nodes: [out_133], Original ATen: [aten.convolution]
        buf610 = extern_kernels.convolution(buf609, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf610, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf611 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_133], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf610, buf611, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf612 = buf604; del buf604  # reuse
        buf613 = buf603; del buf603  # reuse
        buf614 = buf602; del buf602  # reuse
        # Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf611, buf612, buf613, buf614, 3328, 121, grid=grid(3328), stream=stream0)
        buf615 = buf606; del buf606  # reuse
        buf616 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf618 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf612, buf613, buf614, primals_474, primals_475, buf615, buf616, buf618, primals_474, primals_475, 256, 13, grid=grid(256), stream=stream0)
        del primals_474
        del primals_475
        buf619 = reinterpret_tensor(buf610, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf610  # reuse
        # Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf611, buf615, buf616, primals_158, primals_159, buf619, 401408, grid=grid(401408), stream=stream0)
        del primals_159
        # Source Nodes: [out_136], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf621 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_136], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf620, buf621, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf622 = buf594; del buf594  # reuse
        buf623 = buf593; del buf593  # reuse
        buf624 = buf592; del buf592  # reuse
        # Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf621, buf622, buf623, buf624, 6656, 121, grid=grid(6656), stream=stream0)
        buf625 = buf596; del buf596  # reuse
        buf626 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf628 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf622, buf623, buf624, primals_477, primals_478, buf625, buf626, buf628, primals_477, primals_478, 512, 13, grid=grid(512), stream=stream0)
        del primals_477
        del primals_478
        buf629 = reinterpret_tensor(buf632, (8, 512, 14, 14), (301056, 1, 21504, 1536), 0)  # alias
        buf630 = reinterpret_tensor(buf632, (8, 512, 14, 14), (301056, 1, 21504, 1536), 512)  # alias
        buf1179 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_21, out_137, out_138, x2_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_68.run(buf621, buf625, buf626, primals_161, primals_162, buf599, buf629, buf630, buf1179, 802816, grid=grid(802816), stream=stream0)
        del primals_162
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf632, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf634 = reinterpret_tensor(buf620, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf620  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf633, buf634, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf635 = buf624; del buf624  # reuse
        buf636 = buf623; del buf623  # reuse
        buf637 = buf622; del buf622  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf634, buf635, buf636, buf637, 6656, 121, grid=grid(6656), stream=stream0)
        buf638 = buf626; del buf626  # reuse
        buf639 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf641 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf635, buf636, buf637, primals_480, primals_481, buf638, buf639, buf641, primals_480, primals_481, 512, 13, grid=grid(512), stream=stream0)
        del primals_480
        del primals_481
        buf642 = reinterpret_tensor(buf633, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf633  # reuse
        buf778 = empty_strided((8, 2048, 14, 14), (401408, 1, 28672, 2048), device='cuda', dtype=torch.float32)
        buf776 = reinterpret_tensor(buf778, (8, 512, 14, 14), (401408, 1, 28672, 2048), 1024)  # alias
        # Source Nodes: [cat_19, x1_11, x_39, x_40], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_69.run(buf634, buf638, buf639, primals_164, primals_165, buf629, buf642, buf776, 802816, grid=grid(802816), stream=stream0)
        del primals_165
        # Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf643 = extern_kernels.convolution(buf642, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf643, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf644 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_140], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf643, buf644, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf645 = buf614; del buf614  # reuse
        buf646 = buf613; del buf613  # reuse
        buf647 = buf612; del buf612  # reuse
        # Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf644, buf645, buf646, buf647, 3328, 121, grid=grid(3328), stream=stream0)
        buf648 = buf616; del buf616  # reuse
        buf649 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf651 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf645, buf646, buf647, primals_483, primals_484, buf648, buf649, buf651, primals_483, primals_484, 256, 13, grid=grid(256), stream=stream0)
        del primals_483
        del primals_484
        buf652 = reinterpret_tensor(buf643, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf643  # reuse
        # Source Nodes: [out_141, out_142], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf644, buf648, buf649, primals_167, primals_168, buf652, 401408, grid=grid(401408), stream=stream0)
        del primals_168
        # Source Nodes: [out_143], Original ATen: [aten.convolution]
        buf653 = extern_kernels.convolution(buf652, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf653, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf654 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_143], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf653, buf654, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf655 = buf647; del buf647  # reuse
        buf656 = buf646; del buf646  # reuse
        buf657 = buf645; del buf645  # reuse
        # Source Nodes: [out_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf654, buf655, buf656, buf657, 3328, 121, grid=grid(3328), stream=stream0)
        buf658 = buf649; del buf649  # reuse
        buf659 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf661 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf655, buf656, buf657, primals_486, primals_487, buf658, buf659, buf661, primals_486, primals_487, 256, 13, grid=grid(256), stream=stream0)
        del primals_486
        del primals_487
        buf662 = reinterpret_tensor(buf653, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf653  # reuse
        # Source Nodes: [out_144, out_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf654, buf658, buf659, primals_170, primals_171, buf662, 401408, grid=grid(401408), stream=stream0)
        del primals_171
        # Source Nodes: [out_146], Original ATen: [aten.convolution]
        buf663 = extern_kernels.convolution(buf662, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf663, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf664 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_146], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf663, buf664, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf665 = buf637; del buf637  # reuse
        buf666 = buf636; del buf636  # reuse
        buf667 = buf635; del buf635  # reuse
        # Source Nodes: [out_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf664, buf665, buf666, buf667, 6656, 121, grid=grid(6656), stream=stream0)
        buf668 = buf639; del buf639  # reuse
        buf669 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf671 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf665, buf666, buf667, primals_489, primals_490, buf668, buf669, buf671, primals_489, primals_490, 512, 13, grid=grid(512), stream=stream0)
        del primals_489
        del primals_490
        buf672 = reinterpret_tensor(buf663, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf663  # reuse
        # Source Nodes: [out_147, out_148, shortcut_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_70.run(buf664, buf668, buf669, primals_173, primals_174, buf642, buf672, 802816, grid=grid(802816), stream=stream0)
        del primals_174
        # Source Nodes: [out_150], Original ATen: [aten.convolution]
        buf673 = extern_kernels.convolution(buf672, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf673, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf674 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_150], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf673, buf674, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf675 = buf657; del buf657  # reuse
        buf676 = buf656; del buf656  # reuse
        buf677 = buf655; del buf655  # reuse
        # Source Nodes: [out_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf674, buf675, buf676, buf677, 3328, 121, grid=grid(3328), stream=stream0)
        buf678 = buf659; del buf659  # reuse
        buf679 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf681 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf675, buf676, buf677, primals_492, primals_493, buf678, buf679, buf681, primals_492, primals_493, 256, 13, grid=grid(256), stream=stream0)
        del primals_492
        del primals_493
        buf682 = reinterpret_tensor(buf673, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf673  # reuse
        # Source Nodes: [out_151, out_152], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf674, buf678, buf679, primals_176, primals_177, buf682, 401408, grid=grid(401408), stream=stream0)
        del primals_177
        # Source Nodes: [out_153], Original ATen: [aten.convolution]
        buf683 = extern_kernels.convolution(buf682, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf683, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf684 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_153], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf683, buf684, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf685 = buf677; del buf677  # reuse
        buf686 = buf676; del buf676  # reuse
        buf687 = buf675; del buf675  # reuse
        # Source Nodes: [out_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf684, buf685, buf686, buf687, 3328, 121, grid=grid(3328), stream=stream0)
        buf688 = buf679; del buf679  # reuse
        buf689 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf691 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf685, buf686, buf687, primals_495, primals_496, buf688, buf689, buf691, primals_495, primals_496, 256, 13, grid=grid(256), stream=stream0)
        del primals_495
        del primals_496
        buf692 = reinterpret_tensor(buf683, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf683  # reuse
        # Source Nodes: [out_154, out_155], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf684, buf688, buf689, primals_179, primals_180, buf692, 401408, grid=grid(401408), stream=stream0)
        del primals_180
        # Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf693 = extern_kernels.convolution(buf692, primals_181, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf693, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf694 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_156], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf693, buf694, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf695 = buf667; del buf667  # reuse
        buf696 = buf666; del buf666  # reuse
        buf697 = buf665; del buf665  # reuse
        # Source Nodes: [out_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf694, buf695, buf696, buf697, 6656, 121, grid=grid(6656), stream=stream0)
        buf698 = buf669; del buf669  # reuse
        buf699 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf701 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf695, buf696, buf697, primals_498, primals_499, buf698, buf699, buf701, primals_498, primals_499, 512, 13, grid=grid(512), stream=stream0)
        del primals_498
        del primals_499
        buf704 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        buf702 = reinterpret_tensor(buf704, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf703 = reinterpret_tensor(buf704, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        buf1178 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_20, out_157, out_158, x2_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_65.run(buf694, buf698, buf699, primals_182, primals_183, buf672, buf702, buf703, buf1178, 802816, grid=grid(802816), stream=stream0)
        del primals_183
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf705 = extern_kernels.convolution(buf704, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf705, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf706 = reinterpret_tensor(buf693, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf693  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf705, buf706, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf707 = buf697; del buf697  # reuse
        buf708 = buf696; del buf696  # reuse
        buf709 = buf695; del buf695  # reuse
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf706, buf707, buf708, buf709, 6656, 121, grid=grid(6656), stream=stream0)
        buf710 = buf699; del buf699  # reuse
        buf711 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf713 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf707, buf708, buf709, primals_501, primals_502, buf710, buf711, buf713, primals_501, primals_502, 512, 13, grid=grid(512), stream=stream0)
        del primals_501
        del primals_502
        buf714 = reinterpret_tensor(buf705, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf705  # reuse
        # Source Nodes: [x1_13, x_45, x_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_66.run(buf706, buf710, buf711, primals_185, primals_186, buf702, buf714, 802816, grid=grid(802816), stream=stream0)
        del primals_186
        # Source Nodes: [out_160], Original ATen: [aten.convolution]
        buf715 = extern_kernels.convolution(buf714, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf715, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf716 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_160], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf715, buf716, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf717 = buf687; del buf687  # reuse
        buf718 = buf686; del buf686  # reuse
        buf719 = buf685; del buf685  # reuse
        # Source Nodes: [out_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf716, buf717, buf718, buf719, 3328, 121, grid=grid(3328), stream=stream0)
        buf720 = buf689; del buf689  # reuse
        buf721 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf723 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf717, buf718, buf719, primals_504, primals_505, buf720, buf721, buf723, primals_504, primals_505, 256, 13, grid=grid(256), stream=stream0)
        del primals_504
        del primals_505
        buf724 = reinterpret_tensor(buf715, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf715  # reuse
        # Source Nodes: [out_161, out_162], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf716, buf720, buf721, primals_188, primals_189, buf724, 401408, grid=grid(401408), stream=stream0)
        del primals_189
        # Source Nodes: [out_163], Original ATen: [aten.convolution]
        buf725 = extern_kernels.convolution(buf724, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf725, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf726 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_163], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf725, buf726, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf727 = buf719; del buf719  # reuse
        buf728 = buf718; del buf718  # reuse
        buf729 = buf717; del buf717  # reuse
        # Source Nodes: [out_164], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf726, buf727, buf728, buf729, 3328, 121, grid=grid(3328), stream=stream0)
        buf730 = buf721; del buf721  # reuse
        buf731 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf733 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_164], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf727, buf728, buf729, primals_507, primals_508, buf730, buf731, buf733, primals_507, primals_508, 256, 13, grid=grid(256), stream=stream0)
        del primals_507
        del primals_508
        buf734 = reinterpret_tensor(buf725, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf725  # reuse
        # Source Nodes: [out_164, out_165], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf726, buf730, buf731, primals_191, primals_192, buf734, 401408, grid=grid(401408), stream=stream0)
        del primals_192
        # Source Nodes: [out_166], Original ATen: [aten.convolution]
        buf735 = extern_kernels.convolution(buf734, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf735, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf736 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_166], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf735, buf736, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf737 = buf709; del buf709  # reuse
        buf738 = buf708; del buf708  # reuse
        buf739 = buf707; del buf707  # reuse
        # Source Nodes: [out_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf736, buf737, buf738, buf739, 6656, 121, grid=grid(6656), stream=stream0)
        buf740 = buf711; del buf711  # reuse
        buf741 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf743 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf737, buf738, buf739, primals_510, primals_511, buf740, buf741, buf743, primals_510, primals_511, 512, 13, grid=grid(512), stream=stream0)
        del primals_510
        del primals_511
        buf744 = reinterpret_tensor(buf735, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf735  # reuse
        buf777 = reinterpret_tensor(buf778, (8, 512, 14, 14), (401408, 1, 28672, 2048), 1536)  # alias
        # Source Nodes: [cat_19, out_167, out_168, shortcut_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_71.run(buf736, buf740, buf741, primals_194, primals_195, buf714, buf744, buf777, 802816, grid=grid(802816), stream=stream0)
        del primals_195
        # Source Nodes: [out_170], Original ATen: [aten.convolution]
        buf745 = extern_kernels.convolution(buf744, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf745, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf746 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_170], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf745, buf746, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf747 = buf729; del buf729  # reuse
        buf748 = buf728; del buf728  # reuse
        buf749 = buf727; del buf727  # reuse
        # Source Nodes: [out_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf746, buf747, buf748, buf749, 3328, 121, grid=grid(3328), stream=stream0)
        buf750 = buf731; del buf731  # reuse
        buf751 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf753 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf747, buf748, buf749, primals_513, primals_514, buf750, buf751, buf753, primals_513, primals_514, 256, 13, grid=grid(256), stream=stream0)
        del primals_513
        del primals_514
        buf754 = reinterpret_tensor(buf745, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf745  # reuse
        # Source Nodes: [out_171, out_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf746, buf750, buf751, primals_197, primals_198, buf754, 401408, grid=grid(401408), stream=stream0)
        del primals_198
        # Source Nodes: [out_173], Original ATen: [aten.convolution]
        buf755 = extern_kernels.convolution(buf754, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf755, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf756 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_173], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf755, buf756, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf757 = buf749; del buf749  # reuse
        buf758 = buf748; del buf748  # reuse
        buf759 = buf747; del buf747  # reuse
        # Source Nodes: [out_174], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf756, buf757, buf758, buf759, 3328, 121, grid=grid(3328), stream=stream0)
        buf760 = buf751; del buf751  # reuse
        buf761 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf763 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_174], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf757, buf758, buf759, primals_516, primals_517, buf760, buf761, buf763, primals_516, primals_517, 256, 13, grid=grid(256), stream=stream0)
        del primals_516
        del primals_517
        buf764 = reinterpret_tensor(buf755, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf755  # reuse
        # Source Nodes: [out_174, out_175], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf756, buf760, buf761, primals_200, primals_201, buf764, 401408, grid=grid(401408), stream=stream0)
        del primals_201
        # Source Nodes: [out_176], Original ATen: [aten.convolution]
        buf765 = extern_kernels.convolution(buf764, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf765, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf766 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_176], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf765, buf766, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf767 = buf739; del buf739  # reuse
        buf768 = buf738; del buf738  # reuse
        buf769 = buf737; del buf737  # reuse
        # Source Nodes: [out_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf766, buf767, buf768, buf769, 6656, 121, grid=grid(6656), stream=stream0)
        buf770 = buf741; del buf741  # reuse
        buf771 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf773 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf767, buf768, buf769, primals_519, primals_520, buf770, buf771, buf773, primals_519, primals_520, 512, 13, grid=grid(512), stream=stream0)
        del primals_519
        del primals_520
        buf774 = reinterpret_tensor(buf778, (8, 512, 14, 14), (401408, 1, 28672, 2048), 0)  # alias
        buf775 = reinterpret_tensor(buf778, (8, 512, 14, 14), (401408, 1, 28672, 2048), 512)  # alias
        buf1177 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_19, out_177, out_178, x2_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_72.run(buf766, buf770, buf771, primals_203, primals_204, buf744, buf774, buf775, buf1177, 802816, grid=grid(802816), stream=stream0)
        del primals_204
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf779 = extern_kernels.convolution(buf778, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf779, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf780 = reinterpret_tensor(buf765, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf765  # reuse
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf779, buf780, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf781 = buf769; del buf769  # reuse
        buf782 = buf768; del buf768  # reuse
        buf783 = buf767; del buf767  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf780, buf781, buf782, buf783, 6656, 121, grid=grid(6656), stream=stream0)
        buf784 = buf771; del buf771  # reuse
        buf785 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf787 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf781, buf782, buf783, primals_522, primals_523, buf784, buf785, buf787, primals_522, primals_523, 512, 13, grid=grid(512), stream=stream0)
        del primals_522
        del primals_523
        buf788 = reinterpret_tensor(buf779, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf779  # reuse
        buf1068 = reinterpret_tensor(buf1071, (8, 512, 14, 14), (551936, 1, 39424, 2816), 1280)  # alias
        # Source Nodes: [cat_15, x1_15, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_73.run(buf780, buf784, buf785, primals_206, primals_207, buf774, buf788, buf1068, 802816, grid=grid(802816), stream=stream0)
        del primals_207
        # Source Nodes: [out_180], Original ATen: [aten.convolution]
        buf789 = extern_kernels.convolution(buf788, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf789, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf790 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_180], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf789, buf790, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf791 = buf759; del buf759  # reuse
        buf792 = buf758; del buf758  # reuse
        buf793 = buf757; del buf757  # reuse
        # Source Nodes: [out_181], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf790, buf791, buf792, buf793, 3328, 121, grid=grid(3328), stream=stream0)
        buf794 = buf761; del buf761  # reuse
        buf795 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf797 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_181], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf791, buf792, buf793, primals_525, primals_526, buf794, buf795, buf797, primals_525, primals_526, 256, 13, grid=grid(256), stream=stream0)
        del primals_525
        del primals_526
        buf798 = reinterpret_tensor(buf789, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf789  # reuse
        # Source Nodes: [out_181, out_182], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf790, buf794, buf795, primals_209, primals_210, buf798, 401408, grid=grid(401408), stream=stream0)
        del primals_210
        # Source Nodes: [out_183], Original ATen: [aten.convolution]
        buf799 = extern_kernels.convolution(buf798, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf799, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf800 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_183], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf799, buf800, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf801 = buf793; del buf793  # reuse
        buf802 = buf792; del buf792  # reuse
        buf803 = buf791; del buf791  # reuse
        # Source Nodes: [out_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf800, buf801, buf802, buf803, 3328, 121, grid=grid(3328), stream=stream0)
        buf804 = buf795; del buf795  # reuse
        buf805 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf807 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf801, buf802, buf803, primals_528, primals_529, buf804, buf805, buf807, primals_528, primals_529, 256, 13, grid=grid(256), stream=stream0)
        del primals_528
        del primals_529
        buf808 = reinterpret_tensor(buf799, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf799  # reuse
        # Source Nodes: [out_184, out_185], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf800, buf804, buf805, primals_212, primals_213, buf808, 401408, grid=grid(401408), stream=stream0)
        del primals_213
        # Source Nodes: [out_186], Original ATen: [aten.convolution]
        buf809 = extern_kernels.convolution(buf808, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf809, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf810 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_186], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf809, buf810, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf811 = buf783; del buf783  # reuse
        buf812 = buf782; del buf782  # reuse
        buf813 = buf781; del buf781  # reuse
        # Source Nodes: [out_187], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf810, buf811, buf812, buf813, 6656, 121, grid=grid(6656), stream=stream0)
        buf814 = buf785; del buf785  # reuse
        buf815 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf817 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_187], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf811, buf812, buf813, primals_531, primals_532, buf814, buf815, buf817, primals_531, primals_532, 512, 13, grid=grid(512), stream=stream0)
        del primals_531
        del primals_532
        buf818 = reinterpret_tensor(buf809, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf809  # reuse
        # Source Nodes: [out_187, out_188, shortcut_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_70.run(buf810, buf814, buf815, primals_215, primals_216, buf788, buf818, 802816, grid=grid(802816), stream=stream0)
        del primals_216
        # Source Nodes: [out_190], Original ATen: [aten.convolution]
        buf819 = extern_kernels.convolution(buf818, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf819, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf820 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_190], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf819, buf820, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf821 = buf803; del buf803  # reuse
        buf822 = buf802; del buf802  # reuse
        buf823 = buf801; del buf801  # reuse
        # Source Nodes: [out_191], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf820, buf821, buf822, buf823, 3328, 121, grid=grid(3328), stream=stream0)
        buf824 = buf805; del buf805  # reuse
        buf825 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf827 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_191], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf821, buf822, buf823, primals_534, primals_535, buf824, buf825, buf827, primals_534, primals_535, 256, 13, grid=grid(256), stream=stream0)
        del primals_534
        del primals_535
        buf828 = reinterpret_tensor(buf819, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf819  # reuse
        # Source Nodes: [out_191, out_192], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf820, buf824, buf825, primals_218, primals_219, buf828, 401408, grid=grid(401408), stream=stream0)
        del primals_219
        # Source Nodes: [out_193], Original ATen: [aten.convolution]
        buf829 = extern_kernels.convolution(buf828, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf829, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf830 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_193], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf829, buf830, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf831 = buf823; del buf823  # reuse
        buf832 = buf822; del buf822  # reuse
        buf833 = buf821; del buf821  # reuse
        # Source Nodes: [out_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf830, buf831, buf832, buf833, 3328, 121, grid=grid(3328), stream=stream0)
        buf834 = buf825; del buf825  # reuse
        buf835 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf837 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf831, buf832, buf833, primals_537, primals_538, buf834, buf835, buf837, primals_537, primals_538, 256, 13, grid=grid(256), stream=stream0)
        del primals_537
        del primals_538
        buf838 = reinterpret_tensor(buf829, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf829  # reuse
        # Source Nodes: [out_194, out_195], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf830, buf834, buf835, primals_221, primals_222, buf838, 401408, grid=grid(401408), stream=stream0)
        del primals_222
        # Source Nodes: [out_196], Original ATen: [aten.convolution]
        buf839 = extern_kernels.convolution(buf838, primals_223, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf839, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf840 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf839, buf840, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf841 = buf813; del buf813  # reuse
        buf842 = buf812; del buf812  # reuse
        buf843 = buf811; del buf811  # reuse
        # Source Nodes: [out_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf840, buf841, buf842, buf843, 6656, 121, grid=grid(6656), stream=stream0)
        buf844 = buf815; del buf815  # reuse
        buf845 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf847 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf841, buf842, buf843, primals_540, primals_541, buf844, buf845, buf847, primals_540, primals_541, 512, 13, grid=grid(512), stream=stream0)
        del primals_540
        del primals_541
        buf850 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        buf848 = reinterpret_tensor(buf850, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf849 = reinterpret_tensor(buf850, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        buf1176 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_18, out_197, out_198, x2_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_65.run(buf840, buf844, buf845, primals_224, primals_225, buf818, buf848, buf849, buf1176, 802816, grid=grid(802816), stream=stream0)
        del primals_225
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf851 = extern_kernels.convolution(buf850, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf851, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf852 = reinterpret_tensor(buf839, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf839  # reuse
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf851, buf852, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf853 = buf843; del buf843  # reuse
        buf854 = buf842; del buf842  # reuse
        buf855 = buf841; del buf841  # reuse
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf852, buf853, buf854, buf855, 6656, 121, grid=grid(6656), stream=stream0)
        buf856 = buf845; del buf845  # reuse
        buf857 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf859 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf853, buf854, buf855, primals_543, primals_544, buf856, buf857, buf859, primals_543, primals_544, 512, 13, grid=grid(512), stream=stream0)
        del primals_543
        del primals_544
        buf860 = reinterpret_tensor(buf851, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf851  # reuse
        # Source Nodes: [x1_17, x_57, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_66.run(buf852, buf856, buf857, primals_227, primals_228, buf848, buf860, 802816, grid=grid(802816), stream=stream0)
        del primals_228
        # Source Nodes: [out_200], Original ATen: [aten.convolution]
        buf861 = extern_kernels.convolution(buf860, primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf861, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf862 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_200], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf861, buf862, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf863 = buf833; del buf833  # reuse
        buf864 = buf832; del buf832  # reuse
        buf865 = buf831; del buf831  # reuse
        # Source Nodes: [out_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf862, buf863, buf864, buf865, 3328, 121, grid=grid(3328), stream=stream0)
        buf866 = buf835; del buf835  # reuse
        buf867 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf869 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf863, buf864, buf865, primals_546, primals_547, buf866, buf867, buf869, primals_546, primals_547, 256, 13, grid=grid(256), stream=stream0)
        del primals_546
        del primals_547
        buf870 = reinterpret_tensor(buf861, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf861  # reuse
        # Source Nodes: [out_201, out_202], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf862, buf866, buf867, primals_230, primals_231, buf870, 401408, grid=grid(401408), stream=stream0)
        del primals_231
        # Source Nodes: [out_203], Original ATen: [aten.convolution]
        buf871 = extern_kernels.convolution(buf870, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf871, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf872 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_203], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf871, buf872, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf873 = buf865; del buf865  # reuse
        buf874 = buf864; del buf864  # reuse
        buf875 = buf863; del buf863  # reuse
        # Source Nodes: [out_204], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf872, buf873, buf874, buf875, 3328, 121, grid=grid(3328), stream=stream0)
        buf876 = buf867; del buf867  # reuse
        buf877 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf879 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_204], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf873, buf874, buf875, primals_549, primals_550, buf876, buf877, buf879, primals_549, primals_550, 256, 13, grid=grid(256), stream=stream0)
        del primals_549
        del primals_550
        buf880 = reinterpret_tensor(buf871, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf871  # reuse
        # Source Nodes: [out_204, out_205], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf872, buf876, buf877, primals_233, primals_234, buf880, 401408, grid=grid(401408), stream=stream0)
        del primals_234
        # Source Nodes: [out_206], Original ATen: [aten.convolution]
        buf881 = extern_kernels.convolution(buf880, primals_235, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf881, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf882 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_206], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf881, buf882, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf883 = buf855; del buf855  # reuse
        buf884 = buf854; del buf854  # reuse
        buf885 = buf853; del buf853  # reuse
        # Source Nodes: [out_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf882, buf883, buf884, buf885, 6656, 121, grid=grid(6656), stream=stream0)
        buf886 = buf857; del buf857  # reuse
        buf887 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf889 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf883, buf884, buf885, primals_552, primals_553, buf886, buf887, buf889, primals_552, primals_553, 512, 13, grid=grid(512), stream=stream0)
        del primals_552
        del primals_553
        buf890 = reinterpret_tensor(buf881, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf881  # reuse
        buf923 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        buf922 = reinterpret_tensor(buf923, (8, 512, 14, 14), (301056, 1, 21504, 1536), 1024)  # alias
        # Source Nodes: [cat_17, out_207, out_208, shortcut_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_67.run(buf882, buf886, buf887, primals_236, primals_237, buf860, buf890, buf922, 802816, grid=grid(802816), stream=stream0)
        del primals_237
        # Source Nodes: [out_210], Original ATen: [aten.convolution]
        buf891 = extern_kernels.convolution(buf890, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf891, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf892 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_210], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf891, buf892, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf893 = buf875; del buf875  # reuse
        buf894 = buf874; del buf874  # reuse
        buf895 = buf873; del buf873  # reuse
        # Source Nodes: [out_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf892, buf893, buf894, buf895, 3328, 121, grid=grid(3328), stream=stream0)
        buf896 = buf877; del buf877  # reuse
        buf897 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf899 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf893, buf894, buf895, primals_555, primals_556, buf896, buf897, buf899, primals_555, primals_556, 256, 13, grid=grid(256), stream=stream0)
        del primals_555
        del primals_556
        buf900 = reinterpret_tensor(buf891, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf891  # reuse
        # Source Nodes: [out_211, out_212], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf892, buf896, buf897, primals_239, primals_240, buf900, 401408, grid=grid(401408), stream=stream0)
        del primals_240
        # Source Nodes: [out_213], Original ATen: [aten.convolution]
        buf901 = extern_kernels.convolution(buf900, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf901, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf902 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_213], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf901, buf902, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf903 = buf895; del buf895  # reuse
        buf904 = buf894; del buf894  # reuse
        buf905 = buf893; del buf893  # reuse
        # Source Nodes: [out_214], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf902, buf903, buf904, buf905, 3328, 121, grid=grid(3328), stream=stream0)
        buf906 = buf897; del buf897  # reuse
        buf907 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf909 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_214], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf903, buf904, buf905, primals_558, primals_559, buf906, buf907, buf909, primals_558, primals_559, 256, 13, grid=grid(256), stream=stream0)
        del primals_558
        del primals_559
        buf910 = reinterpret_tensor(buf901, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf901  # reuse
        # Source Nodes: [out_214, out_215], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf902, buf906, buf907, primals_242, primals_243, buf910, 401408, grid=grid(401408), stream=stream0)
        del primals_243
        # Source Nodes: [out_216], Original ATen: [aten.convolution]
        buf911 = extern_kernels.convolution(buf910, primals_244, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf911, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf912 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_216], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf911, buf912, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf913 = buf885; del buf885  # reuse
        buf914 = buf884; del buf884  # reuse
        buf915 = buf883; del buf883  # reuse
        # Source Nodes: [out_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf912, buf913, buf914, buf915, 6656, 121, grid=grid(6656), stream=stream0)
        buf916 = buf887; del buf887  # reuse
        buf917 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf919 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf913, buf914, buf915, primals_561, primals_562, buf916, buf917, buf919, primals_561, primals_562, 512, 13, grid=grid(512), stream=stream0)
        del primals_561
        del primals_562
        buf920 = reinterpret_tensor(buf923, (8, 512, 14, 14), (301056, 1, 21504, 1536), 0)  # alias
        buf921 = reinterpret_tensor(buf923, (8, 512, 14, 14), (301056, 1, 21504, 1536), 512)  # alias
        buf1175 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_17, out_217, out_218, x2_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_68.run(buf912, buf916, buf917, primals_245, primals_246, buf890, buf920, buf921, buf1175, 802816, grid=grid(802816), stream=stream0)
        del primals_246
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf924 = extern_kernels.convolution(buf923, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf924, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf925 = reinterpret_tensor(buf911, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf911  # reuse
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf924, buf925, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf926 = buf915; del buf915  # reuse
        buf927 = buf914; del buf914  # reuse
        buf928 = buf913; del buf913  # reuse
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf925, buf926, buf927, buf928, 6656, 121, grid=grid(6656), stream=stream0)
        buf929 = buf917; del buf917  # reuse
        buf930 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf932 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf926, buf927, buf928, primals_564, primals_565, buf929, buf930, buf932, primals_564, primals_565, 512, 13, grid=grid(512), stream=stream0)
        del primals_564
        del primals_565
        buf933 = reinterpret_tensor(buf924, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf924  # reuse
        buf1069 = reinterpret_tensor(buf1071, (8, 512, 14, 14), (551936, 1, 39424, 2816), 1792)  # alias
        # Source Nodes: [cat_15, x1_19, x_62, x_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_74.run(buf925, buf929, buf930, primals_248, primals_249, buf920, buf933, buf1069, 802816, grid=grid(802816), stream=stream0)
        del primals_249
        # Source Nodes: [out_220], Original ATen: [aten.convolution]
        buf934 = extern_kernels.convolution(buf933, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf934, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf935 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf934, buf935, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf936 = buf905; del buf905  # reuse
        buf937 = buf904; del buf904  # reuse
        buf938 = buf903; del buf903  # reuse
        # Source Nodes: [out_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf935, buf936, buf937, buf938, 3328, 121, grid=grid(3328), stream=stream0)
        buf939 = buf907; del buf907  # reuse
        buf940 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf942 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf936, buf937, buf938, primals_567, primals_568, buf939, buf940, buf942, primals_567, primals_568, 256, 13, grid=grid(256), stream=stream0)
        del primals_567
        del primals_568
        buf943 = reinterpret_tensor(buf934, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf934  # reuse
        # Source Nodes: [out_221, out_222], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf935, buf939, buf940, primals_251, primals_252, buf943, 401408, grid=grid(401408), stream=stream0)
        del primals_252
        # Source Nodes: [out_223], Original ATen: [aten.convolution]
        buf944 = extern_kernels.convolution(buf943, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf944, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf945 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_223], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf944, buf945, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf946 = buf938; del buf938  # reuse
        buf947 = buf937; del buf937  # reuse
        buf948 = buf936; del buf936  # reuse
        # Source Nodes: [out_224], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf945, buf946, buf947, buf948, 3328, 121, grid=grid(3328), stream=stream0)
        buf949 = buf940; del buf940  # reuse
        buf950 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf952 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_224], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf946, buf947, buf948, primals_570, primals_571, buf949, buf950, buf952, primals_570, primals_571, 256, 13, grid=grid(256), stream=stream0)
        del primals_570
        del primals_571
        buf953 = reinterpret_tensor(buf944, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf944  # reuse
        # Source Nodes: [out_224, out_225], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf945, buf949, buf950, primals_254, primals_255, buf953, 401408, grid=grid(401408), stream=stream0)
        del primals_255
        # Source Nodes: [out_226], Original ATen: [aten.convolution]
        buf954 = extern_kernels.convolution(buf953, primals_256, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf954, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf955 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_226], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf954, buf955, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf956 = buf928; del buf928  # reuse
        buf957 = buf927; del buf927  # reuse
        buf958 = buf926; del buf926  # reuse
        # Source Nodes: [out_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf955, buf956, buf957, buf958, 6656, 121, grid=grid(6656), stream=stream0)
        buf959 = buf930; del buf930  # reuse
        buf960 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf962 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf956, buf957, buf958, primals_573, primals_574, buf959, buf960, buf962, primals_573, primals_574, 512, 13, grid=grid(512), stream=stream0)
        del primals_573
        del primals_574
        buf963 = reinterpret_tensor(buf954, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf954  # reuse
        # Source Nodes: [out_227, out_228, shortcut_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_70.run(buf955, buf959, buf960, primals_257, primals_258, buf933, buf963, 802816, grid=grid(802816), stream=stream0)
        del primals_258
        # Source Nodes: [out_230], Original ATen: [aten.convolution]
        buf964 = extern_kernels.convolution(buf963, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf964, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf965 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_230], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf964, buf965, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf966 = buf948; del buf948  # reuse
        buf967 = buf947; del buf947  # reuse
        buf968 = buf946; del buf946  # reuse
        # Source Nodes: [out_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf965, buf966, buf967, buf968, 3328, 121, grid=grid(3328), stream=stream0)
        buf969 = buf950; del buf950  # reuse
        buf970 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf972 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf966, buf967, buf968, primals_576, primals_577, buf969, buf970, buf972, primals_576, primals_577, 256, 13, grid=grid(256), stream=stream0)
        del primals_576
        del primals_577
        buf973 = reinterpret_tensor(buf964, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf964  # reuse
        # Source Nodes: [out_231, out_232], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf965, buf969, buf970, primals_260, primals_261, buf973, 401408, grid=grid(401408), stream=stream0)
        del primals_261
        # Source Nodes: [out_233], Original ATen: [aten.convolution]
        buf974 = extern_kernels.convolution(buf973, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf974, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf975 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_233], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf974, buf975, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf976 = buf968; del buf968  # reuse
        buf977 = buf967; del buf967  # reuse
        buf978 = buf966; del buf966  # reuse
        # Source Nodes: [out_234], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf975, buf976, buf977, buf978, 3328, 121, grid=grid(3328), stream=stream0)
        buf979 = buf970; del buf970  # reuse
        buf980 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf982 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_234], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf976, buf977, buf978, primals_579, primals_580, buf979, buf980, buf982, primals_579, primals_580, 256, 13, grid=grid(256), stream=stream0)
        del primals_579
        del primals_580
        buf983 = reinterpret_tensor(buf974, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf974  # reuse
        # Source Nodes: [out_234, out_235], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf975, buf979, buf980, primals_263, primals_264, buf983, 401408, grid=grid(401408), stream=stream0)
        del primals_264
        # Source Nodes: [out_236], Original ATen: [aten.convolution]
        buf984 = extern_kernels.convolution(buf983, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf984, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf985 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_236], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf984, buf985, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf986 = buf958; del buf958  # reuse
        buf987 = buf957; del buf957  # reuse
        buf988 = buf956; del buf956  # reuse
        # Source Nodes: [out_237], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf985, buf986, buf987, buf988, 6656, 121, grid=grid(6656), stream=stream0)
        buf989 = buf960; del buf960  # reuse
        buf990 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf992 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_237], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf986, buf987, buf988, primals_582, primals_583, buf989, buf990, buf992, primals_582, primals_583, 512, 13, grid=grid(512), stream=stream0)
        del primals_582
        del primals_583
        buf995 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        buf993 = reinterpret_tensor(buf995, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf994 = reinterpret_tensor(buf995, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        buf1174 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_16, out_237, out_238, x2_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_65.run(buf985, buf989, buf990, primals_266, primals_267, buf963, buf993, buf994, buf1174, 802816, grid=grid(802816), stream=stream0)
        del primals_267
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf996 = extern_kernels.convolution(buf995, primals_268, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf996, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf997 = reinterpret_tensor(buf984, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf984  # reuse
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf996, buf997, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf998 = buf988; del buf988  # reuse
        buf999 = buf987; del buf987  # reuse
        buf1000 = buf986; del buf986  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf997, buf998, buf999, buf1000, 6656, 121, grid=grid(6656), stream=stream0)
        buf1001 = buf990; del buf990  # reuse
        buf1002 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf1004 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf998, buf999, buf1000, primals_585, primals_586, buf1001, buf1002, buf1004, primals_585, primals_586, 512, 13, grid=grid(512), stream=stream0)
        del primals_585
        del primals_586
        buf1005 = reinterpret_tensor(buf996, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf996  # reuse
        # Source Nodes: [x1_21, x_68, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_66.run(buf997, buf1001, buf1002, primals_269, primals_270, buf993, buf1005, 802816, grid=grid(802816), stream=stream0)
        del primals_270
        # Source Nodes: [out_240], Original ATen: [aten.convolution]
        buf1006 = extern_kernels.convolution(buf1005, primals_271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1006, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf1007 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_240], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf1006, buf1007, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf1008 = buf978; del buf978  # reuse
        buf1009 = buf977; del buf977  # reuse
        buf1010 = buf976; del buf976  # reuse
        # Source Nodes: [out_241], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf1007, buf1008, buf1009, buf1010, 3328, 121, grid=grid(3328), stream=stream0)
        buf1011 = buf980; del buf980  # reuse
        buf1012 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf1014 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_241], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf1008, buf1009, buf1010, primals_588, primals_589, buf1011, buf1012, buf1014, primals_588, primals_589, 256, 13, grid=grid(256), stream=stream0)
        del primals_588
        del primals_589
        buf1015 = reinterpret_tensor(buf1006, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1006  # reuse
        # Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf1007, buf1011, buf1012, primals_272, primals_273, buf1015, 401408, grid=grid(401408), stream=stream0)
        del primals_273
        # Source Nodes: [out_243], Original ATen: [aten.convolution]
        buf1016 = extern_kernels.convolution(buf1015, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1016, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf1017 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_243], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf1016, buf1017, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf1018 = buf1010; del buf1010  # reuse
        buf1019 = buf1009; del buf1009  # reuse
        buf1020 = buf1008; del buf1008  # reuse
        # Source Nodes: [out_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf1017, buf1018, buf1019, buf1020, 3328, 121, grid=grid(3328), stream=stream0)
        buf1021 = buf1012; del buf1012  # reuse
        buf1022 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf1024 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf1018, buf1019, buf1020, primals_591, primals_592, buf1021, buf1022, buf1024, primals_591, primals_592, 256, 13, grid=grid(256), stream=stream0)
        del primals_591
        del primals_592
        buf1025 = reinterpret_tensor(buf1016, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1016  # reuse
        # Source Nodes: [out_244, out_245], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf1017, buf1021, buf1022, primals_275, primals_276, buf1025, 401408, grid=grid(401408), stream=stream0)
        del primals_276
        # Source Nodes: [out_246], Original ATen: [aten.convolution]
        buf1026 = extern_kernels.convolution(buf1025, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1026, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf1027 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_246], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf1026, buf1027, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf1028 = buf999; del buf999  # reuse
        buf1029 = buf998; del buf998  # reuse
        buf1030 = buf1000; del buf1000  # reuse
        # Source Nodes: [out_247], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf1027, buf1028, buf1029, buf1030, 6656, 121, grid=grid(6656), stream=stream0)
        buf1031 = buf1002; del buf1002  # reuse
        buf1032 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf1034 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_247], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf1028, buf1029, buf1030, primals_594, primals_595, buf1031, buf1032, buf1034, primals_594, primals_595, 512, 13, grid=grid(512), stream=stream0)
        del primals_594
        del primals_595
        buf1035 = reinterpret_tensor(buf1026, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf1026  # reuse
        buf1070 = reinterpret_tensor(buf1071, (8, 512, 14, 14), (551936, 1, 39424, 2816), 2304)  # alias
        # Source Nodes: [cat_15, out_247, out_248, shortcut_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_75.run(buf1027, buf1031, buf1032, primals_278, primals_279, buf1005, buf1035, buf1070, 802816, grid=grid(802816), stream=stream0)
        del primals_279
        # Source Nodes: [out_250], Original ATen: [aten.convolution]
        buf1036 = extern_kernels.convolution(buf1035, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1036, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf1037 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_250], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf1036, buf1037, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf1038 = buf1020; del buf1020  # reuse
        buf1039 = buf1019; del buf1019  # reuse
        buf1040 = buf1018; del buf1018  # reuse
        # Source Nodes: [out_251], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf1037, buf1038, buf1039, buf1040, 3328, 121, grid=grid(3328), stream=stream0)
        buf1041 = buf1022; del buf1022  # reuse
        buf1042 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf1044 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_251], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf1038, buf1039, buf1040, primals_597, primals_598, buf1041, buf1042, buf1044, primals_597, primals_598, 256, 13, grid=grid(256), stream=stream0)
        del primals_597
        del primals_598
        buf1045 = reinterpret_tensor(buf1036, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1036  # reuse
        # Source Nodes: [out_251, out_252], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf1037, buf1041, buf1042, primals_281, primals_282, buf1045, 401408, grid=grid(401408), stream=stream0)
        del primals_282
        # Source Nodes: [out_253], Original ATen: [aten.convolution]
        buf1046 = extern_kernels.convolution(buf1045, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1046, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf1047 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_253], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf1046, buf1047, 2048, 196, grid=grid(2048, 196), stream=stream0)
        buf1048 = buf1040; del buf1040  # reuse
        buf1049 = buf1039; del buf1039  # reuse
        buf1050 = buf1038; del buf1038  # reuse
        # Source Nodes: [out_254], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf1047, buf1048, buf1049, buf1050, 3328, 121, grid=grid(3328), stream=stream0)
        buf1051 = buf1042; del buf1042  # reuse
        buf1052 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf1054 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_254], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf1048, buf1049, buf1050, primals_600, primals_601, buf1051, buf1052, buf1054, primals_600, primals_601, 256, 13, grid=grid(256), stream=stream0)
        del buf1048
        del buf1049
        del buf1050
        del primals_600
        del primals_601
        buf1055 = reinterpret_tensor(buf1046, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1046  # reuse
        # Source Nodes: [out_254, out_255], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_63.run(buf1047, buf1051, buf1052, primals_284, primals_285, buf1055, 401408, grid=grid(401408), stream=stream0)
        del buf1052
        del primals_285
        # Source Nodes: [out_256], Original ATen: [aten.convolution]
        buf1056 = extern_kernels.convolution(buf1055, primals_286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1056, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf1057 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_256], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf1056, buf1057, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf1058 = buf1030; del buf1030  # reuse
        buf1059 = buf1029; del buf1029  # reuse
        buf1060 = buf1028; del buf1028  # reuse
        # Source Nodes: [out_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf1057, buf1058, buf1059, buf1060, 6656, 121, grid=grid(6656), stream=stream0)
        buf1061 = buf1032; del buf1032  # reuse
        buf1062 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf1064 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf1058, buf1059, buf1060, primals_603, primals_604, buf1061, buf1062, buf1064, primals_603, primals_604, 512, 13, grid=grid(512), stream=stream0)
        del primals_603
        del primals_604
        buf1065 = reinterpret_tensor(buf1071, (8, 512, 14, 14), (551936, 1, 39424, 2816), 0)  # alias
        buf1066 = reinterpret_tensor(buf1071, (8, 512, 14, 14), (551936, 1, 39424, 2816), 512)  # alias
        buf1173 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [cat_15, out_257, out_258, x2_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_threshold_backward_76.run(buf1057, buf1061, buf1062, primals_287, primals_288, buf1035, buf1065, buf1066, buf1173, 802816, grid=grid(802816), stream=stream0)
        del primals_288
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf1072 = extern_kernels.convolution(buf1071, primals_289, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1072, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf1073 = reinterpret_tensor(buf1056, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf1056  # reuse
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf1072, buf1073, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf1074 = buf1060; del buf1060  # reuse
        buf1075 = buf1059; del buf1059  # reuse
        buf1076 = buf1058; del buf1058  # reuse
        # Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf1073, buf1074, buf1075, buf1076, 6656, 121, grid=grid(6656), stream=stream0)
        buf1077 = buf1062; del buf1062  # reuse
        buf1078 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf1080 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf1074, buf1075, buf1076, primals_606, primals_607, buf1077, buf1078, buf1080, primals_606, primals_607, 512, 13, grid=grid(512), stream=stream0)
        del primals_606
        del primals_607
        buf1081 = reinterpret_tensor(buf1072, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf1072  # reuse
        # Source Nodes: [x_73, x_74, x_80], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_77.run(buf1073, buf1077, buf1078, primals_290, primals_291, buf1065, buf1081, 802816, grid=grid(802816), stream=stream0)
        del primals_291
        buf1082 = empty_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        buf1083 = empty_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.int64)
        buf1156 = empty_strided((8, 2560, 7, 7), (125440, 1, 17920, 2560), device='cuda', dtype=torch.float32)
        buf1155 = reinterpret_tensor(buf1156, (8, 512, 7, 7), (125440, 1, 17920, 2560), 2048)  # alias
        # Source Nodes: [bottom_23, cat_14], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
        triton_poi_fused_cat_max_pool2d_with_indices_78.run(buf1081, buf1082, buf1083, buf1155, 200704, grid=grid(200704), stream=stream0)
        # Source Nodes: [l__mod___level5_project_0], Original ATen: [aten.convolution]
        buf1084 = extern_kernels.convolution(buf1082, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1084, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf1085 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level5_project_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1084, buf1085, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf1086 = empty_strided((1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), device='cuda', dtype=torch.float32)
        buf1087 = empty_strided((1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), device='cuda', dtype=torch.float32)
        buf1088 = empty_strided((1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1085, buf1086, buf1087, buf1088, 4096, 98, grid=grid(4096), stream=stream0)
        buf1089 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1090 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1092 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1086, buf1087, buf1088, primals_609, primals_610, buf1089, buf1090, buf1092, primals_609, primals_610, 1024, 4, grid=grid(1024), stream=stream0)
        del primals_609
        del primals_610
        # Source Nodes: [out_260], Original ATen: [aten.convolution]
        buf1093 = extern_kernels.convolution(buf1081, primals_295, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1093, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf1094 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_260], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf1093, buf1094, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf1095 = buf1076; del buf1076  # reuse
        buf1096 = buf1075; del buf1075  # reuse
        buf1097 = buf1074; del buf1074  # reuse
        # Source Nodes: [out_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf1094, buf1095, buf1096, buf1097, 6656, 121, grid=grid(6656), stream=stream0)
        buf1098 = buf1078; del buf1078  # reuse
        buf1099 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf1101 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf1095, buf1096, buf1097, primals_612, primals_613, buf1098, buf1099, buf1101, primals_612, primals_613, 512, 13, grid=grid(512), stream=stream0)
        del buf1095
        del buf1096
        del buf1097
        del primals_612
        del primals_613
        buf1102 = reinterpret_tensor(buf1093, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf1093  # reuse
        # Source Nodes: [out_261, out_262], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_82.run(buf1094, buf1098, buf1099, primals_296, primals_297, buf1102, 802816, grid=grid(802816), stream=stream0)
        del primals_297
        # Source Nodes: [out_263], Original ATen: [aten.convolution]
        buf1103 = extern_kernels.convolution(buf1102, buf29, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1103, (8, 512, 7, 7), (25088, 49, 7, 1))
        buf1104 = empty_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_263], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf1103, buf1104, 4096, 49, grid=grid(4096, 49), stream=stream0)
        buf1105 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        buf1106 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        buf1107 = empty_strided((1, 512, 1, 1, 4), (2048, 1, 2048, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_264], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf1104, buf1105, buf1106, buf1107, 2048, 98, grid=grid(2048), stream=stream0)
        buf1108 = buf1099; del buf1099  # reuse
        buf1109 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf1111 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_264], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf1105, buf1106, buf1107, primals_615, primals_616, buf1108, buf1109, buf1111, primals_615, primals_616, 512, 4, grid=grid(512), stream=stream0)
        del primals_615
        del primals_616
        buf1112 = reinterpret_tensor(buf1103, (8, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1103  # reuse
        # Source Nodes: [out_264, out_265], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_86.run(buf1104, buf1108, buf1109, primals_299, primals_300, buf1112, 200704, grid=grid(200704), stream=stream0)
        del primals_300
        # Source Nodes: [out_266], Original ATen: [aten.convolution]
        buf1113 = extern_kernels.convolution(buf1112, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1113, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf1114 = reinterpret_tensor(buf1084, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf1084  # reuse
        # Source Nodes: [out_266], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1113, buf1114, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf1115 = buf1088; del buf1088  # reuse
        buf1116 = buf1087; del buf1087  # reuse
        buf1117 = buf1086; del buf1086  # reuse
        # Source Nodes: [out_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1114, buf1115, buf1116, buf1117, 4096, 98, grid=grid(4096), stream=stream0)
        buf1118 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1119 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1121 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1115, buf1116, buf1117, primals_618, primals_619, buf1118, buf1119, buf1121, primals_618, primals_619, 1024, 4, grid=grid(1024), stream=stream0)
        del primals_618
        del primals_619
        buf1122 = reinterpret_tensor(buf1113, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf1113  # reuse
        buf1123 = buf1122; del buf1122  # reuse
        # Source Nodes: [out_267, out_268, shortcut_36, shortcut_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_87.run(buf1123, buf1114, buf1118, buf1119, primals_302, primals_303, buf1085, buf1089, buf1090, primals_293, primals_294, 401408, grid=grid(401408), stream=stream0)
        del primals_294
        del primals_303
        # Source Nodes: [out_270], Original ATen: [aten.convolution]
        buf1124 = extern_kernels.convolution(buf1123, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1124, (8, 512, 7, 7), (25088, 49, 7, 1))
        buf1125 = empty_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_270], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf1124, buf1125, 4096, 49, grid=grid(4096, 49), stream=stream0)
        buf1126 = buf1107; del buf1107  # reuse
        buf1127 = buf1106; del buf1106  # reuse
        buf1128 = buf1105; del buf1105  # reuse
        # Source Nodes: [out_271], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf1125, buf1126, buf1127, buf1128, 2048, 98, grid=grid(2048), stream=stream0)
        buf1129 = buf1109; del buf1109  # reuse
        buf1130 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf1132 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_271], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf1126, buf1127, buf1128, primals_621, primals_622, buf1129, buf1130, buf1132, primals_621, primals_622, 512, 4, grid=grid(512), stream=stream0)
        del primals_621
        del primals_622
        buf1133 = reinterpret_tensor(buf1124, (8, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1124  # reuse
        # Source Nodes: [out_271, out_272], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_86.run(buf1125, buf1129, buf1130, primals_305, primals_306, buf1133, 200704, grid=grid(200704), stream=stream0)
        del primals_306
        # Source Nodes: [out_273], Original ATen: [aten.convolution]
        buf1134 = extern_kernels.convolution(buf1133, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1134, (8, 512, 7, 7), (25088, 49, 7, 1))
        buf1135 = empty_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_273], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf1134, buf1135, 4096, 49, grid=grid(4096, 49), stream=stream0)
        buf1136 = buf1128; del buf1128  # reuse
        buf1137 = buf1127; del buf1127  # reuse
        buf1138 = buf1126; del buf1126  # reuse
        # Source Nodes: [out_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf1135, buf1136, buf1137, buf1138, 2048, 98, grid=grid(2048), stream=stream0)
        buf1139 = buf1130; del buf1130  # reuse
        buf1140 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf1142 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf1136, buf1137, buf1138, primals_624, primals_625, buf1139, buf1140, buf1142, primals_624, primals_625, 512, 4, grid=grid(512), stream=stream0)
        del buf1136
        del buf1137
        del buf1138
        del primals_624
        del primals_625
        buf1143 = reinterpret_tensor(buf1134, (8, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1134  # reuse
        # Source Nodes: [out_274, out_275], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_86.run(buf1135, buf1139, buf1140, primals_308, primals_309, buf1143, 200704, grid=grid(200704), stream=stream0)
        del buf1140
        del primals_309
        # Source Nodes: [out_276], Original ATen: [aten.convolution]
        buf1144 = extern_kernels.convolution(buf1143, primals_310, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1144, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf1145 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_276], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1144, buf1145, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf1146 = buf1117; del buf1117  # reuse
        buf1147 = buf1116; del buf1116  # reuse
        buf1148 = buf1115; del buf1115  # reuse
        # Source Nodes: [out_277], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1145, buf1146, buf1147, buf1148, 4096, 98, grid=grid(4096), stream=stream0)
        buf1149 = buf1119; del buf1119  # reuse
        buf1150 = buf1090; del buf1090  # reuse
        buf1152 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_277], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1146, buf1147, buf1148, primals_627, primals_628, buf1149, buf1150, buf1152, primals_627, primals_628, 1024, 4, grid=grid(1024), stream=stream0)
        del primals_627
        del primals_628
        buf1153 = reinterpret_tensor(buf1156, (8, 1024, 7, 7), (125440, 1, 17920, 2560), 0)  # alias
        buf1154 = reinterpret_tensor(buf1156, (8, 1024, 7, 7), (125440, 1, 17920, 2560), 1024)  # alias
        # Source Nodes: [cat_14, out_277, out_278, x2_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_relu_88.run(buf1145, buf1149, buf1150, primals_311, primals_312, buf1123, buf1153, buf1154, 401408, grid=grid(401408), stream=stream0)
        del primals_312
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf1157 = extern_kernels.convolution(buf1156, primals_313, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1157, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf1158 = reinterpret_tensor(buf1144, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf1144  # reuse
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf1157, buf1158, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf1159 = buf1148; del buf1148  # reuse
        buf1160 = buf1147; del buf1147  # reuse
        buf1161 = buf1146; del buf1146  # reuse
        # Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf1158, buf1159, buf1160, buf1161, 4096, 98, grid=grid(4096), stream=stream0)
        buf1162 = buf1150; del buf1150  # reuse
        buf1163 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1165 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf1159, buf1160, buf1161, primals_630, primals_631, buf1162, buf1163, buf1165, primals_630, primals_631, 1024, 4, grid=grid(1024), stream=stream0)
        del buf1159
        del buf1160
        del buf1161
        del primals_630
        del primals_631
        buf1166 = reinterpret_tensor(buf1157, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf1157  # reuse
        buf1171 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.bool)
        buf1172 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_82, x_83, x_87], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_89.run(buf1158, buf1162, buf1163, primals_314, primals_315, buf1153, buf1166, buf1171, buf1172, 401408, grid=grid(401408), stream=stream0)
        del buf1163
        del primals_315
        buf1167 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf1168 = reinterpret_tensor(buf1167, (8, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf1167  # reuse
        # Source Nodes: [x_88], Original ATen: [aten.mean]
        triton_per_fused_mean_90.run(buf1168, buf1166, 8192, 49, grid=grid(8192), stream=stream0)
        del buf1166
        # Source Nodes: [x_92], Original ATen: [aten.convolution]
        buf1169 = extern_kernels.convolution(buf1168, primals_316, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1169, (8, 1000, 1, 1), (1000, 1, 1, 1))
        buf1170 = reinterpret_tensor(buf1169, (8, 1000), (1000, 1), 0); del buf1169  # reuse
        # Source Nodes: [pred, x_92], Original ATen: [aten.convolution, aten.view]
        triton_poi_fused_convolution_view_91.run(buf1170, primals_317, 8000, grid=grid(8000), stream=stream0)
        del primals_317
        # Source Nodes: [l__mod___base_layer_1], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_320, primals_320, 1, grid=grid(1), stream=stream0)
        del primals_320
        # Source Nodes: [l__mod___level0_1], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_323, primals_323, 1, grid=grid(1), stream=stream0)
        del primals_323
        # Source Nodes: [l__mod___level1_1], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_326, primals_326, 1, grid=grid(1), stream=stream0)
        del primals_326
        # Source Nodes: [shortcut], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_329, primals_329, 1, grid=grid(1), stream=stream0)
        del primals_329
        # Source Nodes: [out_1], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_332, primals_332, 1, grid=grid(1), stream=stream0)
        del primals_332
        # Source Nodes: [out_4], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_335, primals_335, 1, grid=grid(1), stream=stream0)
        del primals_335
        # Source Nodes: [out_7], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_338, primals_338, 1, grid=grid(1), stream=stream0)
        del primals_338
        # Source Nodes: [out_11], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_341, primals_341, 1, grid=grid(1), stream=stream0)
        del primals_341
        # Source Nodes: [out_14], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_344, primals_344, 1, grid=grid(1), stream=stream0)
        del primals_344
        # Source Nodes: [out_17], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_347, primals_347, 1, grid=grid(1), stream=stream0)
        del primals_347
        # Source Nodes: [x_4], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_350, primals_350, 1, grid=grid(1), stream=stream0)
        del primals_350
        # Source Nodes: [shortcut_4], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_353, primals_353, 1, grid=grid(1), stream=stream0)
        del primals_353
        # Source Nodes: [out_21], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_356, primals_356, 1, grid=grid(1), stream=stream0)
        del primals_356
        # Source Nodes: [out_24], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_359, primals_359, 1, grid=grid(1), stream=stream0)
        del primals_359
        # Source Nodes: [out_27], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_362, primals_362, 1, grid=grid(1), stream=stream0)
        del primals_362
        # Source Nodes: [out_31], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_365, primals_365, 1, grid=grid(1), stream=stream0)
        del primals_365
        # Source Nodes: [out_34], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_368, primals_368, 1, grid=grid(1), stream=stream0)
        del primals_368
        # Source Nodes: [out_37], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_371, primals_371, 1, grid=grid(1), stream=stream0)
        del primals_371
        # Source Nodes: [x_10], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_374, primals_374, 1, grid=grid(1), stream=stream0)
        del primals_374
        # Source Nodes: [out_41], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_377, primals_377, 1, grid=grid(1), stream=stream0)
        del primals_377
        # Source Nodes: [out_44], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_380, primals_380, 1, grid=grid(1), stream=stream0)
        del primals_380
        # Source Nodes: [out_47], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_383, primals_383, 1, grid=grid(1), stream=stream0)
        del primals_383
        # Source Nodes: [out_51], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_386, primals_386, 1, grid=grid(1), stream=stream0)
        del primals_386
        # Source Nodes: [out_54], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_389, primals_389, 1, grid=grid(1), stream=stream0)
        del primals_389
        # Source Nodes: [out_57], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_392, primals_392, 1, grid=grid(1), stream=stream0)
        del primals_392
        # Source Nodes: [x_15], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_395, primals_395, 1, grid=grid(1), stream=stream0)
        del primals_395
        # Source Nodes: [out_61], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_398, primals_398, 1, grid=grid(1), stream=stream0)
        del primals_398
        # Source Nodes: [out_64], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_401, primals_401, 1, grid=grid(1), stream=stream0)
        del primals_401
        # Source Nodes: [out_67], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_404, primals_404, 1, grid=grid(1), stream=stream0)
        del primals_404
        # Source Nodes: [out_71], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_407, primals_407, 1, grid=grid(1), stream=stream0)
        del primals_407
        # Source Nodes: [out_74], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_410, primals_410, 1, grid=grid(1), stream=stream0)
        del primals_410
        # Source Nodes: [out_77], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_413, primals_413, 1, grid=grid(1), stream=stream0)
        del primals_413
        # Source Nodes: [x_21], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_416, primals_416, 1, grid=grid(1), stream=stream0)
        del primals_416
        # Source Nodes: [out_81], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_419, primals_419, 1, grid=grid(1), stream=stream0)
        del primals_419
        # Source Nodes: [out_84], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_422, primals_422, 1, grid=grid(1), stream=stream0)
        del primals_422
        # Source Nodes: [out_87], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_425, primals_425, 1, grid=grid(1), stream=stream0)
        del primals_425
        # Source Nodes: [out_91], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_428, primals_428, 1, grid=grid(1), stream=stream0)
        del primals_428
        # Source Nodes: [out_94], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_431, primals_431, 1, grid=grid(1), stream=stream0)
        del primals_431
        # Source Nodes: [out_97], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_434, primals_434, 1, grid=grid(1), stream=stream0)
        del primals_434
        # Source Nodes: [x_26], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_437, primals_437, 1, grid=grid(1), stream=stream0)
        del primals_437
        # Source Nodes: [shortcut_16], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_440, primals_440, 1, grid=grid(1), stream=stream0)
        del primals_440
        # Source Nodes: [out_101], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_443, primals_443, 1, grid=grid(1), stream=stream0)
        del primals_443
        # Source Nodes: [out_104], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_446, primals_446, 1, grid=grid(1), stream=stream0)
        del primals_446
        # Source Nodes: [out_107], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_449, primals_449, 1, grid=grid(1), stream=stream0)
        del primals_449
        # Source Nodes: [out_111], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_452, primals_452, 1, grid=grid(1), stream=stream0)
        del primals_452
        # Source Nodes: [out_114], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_455, primals_455, 1, grid=grid(1), stream=stream0)
        del primals_455
        # Source Nodes: [out_117], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_458, primals_458, 1, grid=grid(1), stream=stream0)
        del primals_458
        # Source Nodes: [x_34], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_461, primals_461, 1, grid=grid(1), stream=stream0)
        del primals_461
        # Source Nodes: [out_121], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_464, primals_464, 1, grid=grid(1), stream=stream0)
        del primals_464
        # Source Nodes: [out_124], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_467, primals_467, 1, grid=grid(1), stream=stream0)
        del primals_467
        # Source Nodes: [out_127], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_470, primals_470, 1, grid=grid(1), stream=stream0)
        del primals_470
        # Source Nodes: [out_131], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_473, primals_473, 1, grid=grid(1), stream=stream0)
        del primals_473
        # Source Nodes: [out_134], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_476, primals_476, 1, grid=grid(1), stream=stream0)
        del primals_476
        # Source Nodes: [out_137], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_479, primals_479, 1, grid=grid(1), stream=stream0)
        del primals_479
        # Source Nodes: [x_39], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_482, primals_482, 1, grid=grid(1), stream=stream0)
        del primals_482
        # Source Nodes: [out_141], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_485, primals_485, 1, grid=grid(1), stream=stream0)
        del primals_485
        # Source Nodes: [out_144], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_488, primals_488, 1, grid=grid(1), stream=stream0)
        del primals_488
        # Source Nodes: [out_147], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_491, primals_491, 1, grid=grid(1), stream=stream0)
        del primals_491
        # Source Nodes: [out_151], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_494, primals_494, 1, grid=grid(1), stream=stream0)
        del primals_494
        # Source Nodes: [out_154], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_497, primals_497, 1, grid=grid(1), stream=stream0)
        del primals_497
        # Source Nodes: [out_157], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_500, primals_500, 1, grid=grid(1), stream=stream0)
        del primals_500
        # Source Nodes: [x_45], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_503, primals_503, 1, grid=grid(1), stream=stream0)
        del primals_503
        # Source Nodes: [out_161], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_506, primals_506, 1, grid=grid(1), stream=stream0)
        del primals_506
        # Source Nodes: [out_164], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_509, primals_509, 1, grid=grid(1), stream=stream0)
        del primals_509
        # Source Nodes: [out_167], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_512, primals_512, 1, grid=grid(1), stream=stream0)
        del primals_512
        # Source Nodes: [out_171], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_515, primals_515, 1, grid=grid(1), stream=stream0)
        del primals_515
        # Source Nodes: [out_174], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_518, primals_518, 1, grid=grid(1), stream=stream0)
        del primals_518
        # Source Nodes: [out_177], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_521, primals_521, 1, grid=grid(1), stream=stream0)
        del primals_521
        # Source Nodes: [x_50], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_524, primals_524, 1, grid=grid(1), stream=stream0)
        del primals_524
        # Source Nodes: [out_181], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_527, primals_527, 1, grid=grid(1), stream=stream0)
        del primals_527
        # Source Nodes: [out_184], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_530, primals_530, 1, grid=grid(1), stream=stream0)
        del primals_530
        # Source Nodes: [out_187], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_533, primals_533, 1, grid=grid(1), stream=stream0)
        del primals_533
        # Source Nodes: [out_191], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_536, primals_536, 1, grid=grid(1), stream=stream0)
        del primals_536
        # Source Nodes: [out_194], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_539, primals_539, 1, grid=grid(1), stream=stream0)
        del primals_539
        # Source Nodes: [out_197], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_542, primals_542, 1, grid=grid(1), stream=stream0)
        del primals_542
        # Source Nodes: [x_57], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_545, primals_545, 1, grid=grid(1), stream=stream0)
        del primals_545
        # Source Nodes: [out_201], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_548, primals_548, 1, grid=grid(1), stream=stream0)
        del primals_548
        # Source Nodes: [out_204], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_551, primals_551, 1, grid=grid(1), stream=stream0)
        del primals_551
        # Source Nodes: [out_207], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_554, primals_554, 1, grid=grid(1), stream=stream0)
        del primals_554
        # Source Nodes: [out_211], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_557, primals_557, 1, grid=grid(1), stream=stream0)
        del primals_557
        # Source Nodes: [out_214], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_560, primals_560, 1, grid=grid(1), stream=stream0)
        del primals_560
        # Source Nodes: [out_217], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_563, primals_563, 1, grid=grid(1), stream=stream0)
        del primals_563
        # Source Nodes: [x_62], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_566, primals_566, 1, grid=grid(1), stream=stream0)
        del primals_566
        # Source Nodes: [out_221], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_569, primals_569, 1, grid=grid(1), stream=stream0)
        del primals_569
        # Source Nodes: [out_224], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_572, primals_572, 1, grid=grid(1), stream=stream0)
        del primals_572
        # Source Nodes: [out_227], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_575, primals_575, 1, grid=grid(1), stream=stream0)
        del primals_575
        # Source Nodes: [out_231], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_578, primals_578, 1, grid=grid(1), stream=stream0)
        del primals_578
        # Source Nodes: [out_234], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_581, primals_581, 1, grid=grid(1), stream=stream0)
        del primals_581
        # Source Nodes: [out_237], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_584, primals_584, 1, grid=grid(1), stream=stream0)
        del primals_584
        # Source Nodes: [x_68], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_587, primals_587, 1, grid=grid(1), stream=stream0)
        del primals_587
        # Source Nodes: [out_241], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_590, primals_590, 1, grid=grid(1), stream=stream0)
        del primals_590
        # Source Nodes: [out_244], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_593, primals_593, 1, grid=grid(1), stream=stream0)
        del primals_593
        # Source Nodes: [out_247], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_596, primals_596, 1, grid=grid(1), stream=stream0)
        del primals_596
        # Source Nodes: [out_251], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_599, primals_599, 1, grid=grid(1), stream=stream0)
        del primals_599
        # Source Nodes: [out_254], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_602, primals_602, 1, grid=grid(1), stream=stream0)
        del primals_602
        # Source Nodes: [out_257], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_605, primals_605, 1, grid=grid(1), stream=stream0)
        del primals_605
        # Source Nodes: [x_73], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_608, primals_608, 1, grid=grid(1), stream=stream0)
        del primals_608
        # Source Nodes: [shortcut_36], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_611, primals_611, 1, grid=grid(1), stream=stream0)
        del primals_611
        # Source Nodes: [out_261], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_614, primals_614, 1, grid=grid(1), stream=stream0)
        del primals_614
        # Source Nodes: [out_264], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_617, primals_617, 1, grid=grid(1), stream=stream0)
        del primals_617
        # Source Nodes: [out_267], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_620, primals_620, 1, grid=grid(1), stream=stream0)
        del primals_620
        # Source Nodes: [out_271], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_623, primals_623, 1, grid=grid(1), stream=stream0)
        del primals_623
        # Source Nodes: [out_274], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_626, primals_626, 1, grid=grid(1), stream=stream0)
        del primals_626
        # Source Nodes: [out_277], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_629, primals_629, 1, grid=grid(1), stream=stream0)
        del primals_629
        # Source Nodes: [x_82], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(primals_632, primals_632, 1, grid=grid(1), stream=stream0)
        del primals_632
        return (buf1170, buf0, primals_2, buf1, primals_5, buf2, primals_8, primals_10, primals_11, primals_13, primals_14, buf3, primals_17, primals_19, primals_20, primals_22, primals_23, buf4, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, buf5, primals_41, primals_43, primals_44, primals_46, primals_47, buf6, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, buf7, primals_62, primals_64, primals_65, primals_67, primals_68, buf8, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, buf9, primals_83, primals_85, primals_86, primals_88, primals_89, buf10, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, buf11, primals_104, primals_106, primals_107, primals_109, primals_110, buf12, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, buf13, primals_128, primals_130, primals_131, primals_133, primals_134, buf14, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, buf15, primals_149, primals_151, primals_152, primals_154, primals_155, buf16, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, buf17, primals_170, primals_172, primals_173, primals_175, primals_176, buf18, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, buf19, primals_191, primals_193, primals_194, primals_196, primals_197, buf20, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, buf21, primals_212, primals_214, primals_215, primals_217, primals_218, buf22, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, buf23, primals_233, primals_235, primals_236, primals_238, primals_239, buf24, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, buf25, primals_254, primals_256, primals_257, primals_259, primals_260, buf26, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, buf27, primals_275, primals_277, primals_278, primals_280, primals_281, buf28, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, buf29, primals_299, primals_301, primals_302, primals_304, primals_305, buf30, primals_308, primals_310, primals_311, primals_313, primals_314, primals_316, buf31, buf33, buf43, buf44, buf46, buf56, buf57, buf59, buf69, buf70, buf71, buf72, buf74, buf84, buf86, buf96, buf97, buf99, buf109, buf110, buf112, buf122, buf124, buf126, buf136, buf137, buf139, buf149, buf150, buf152, buf162, buf165, buf167, buf177, buf178, buf179, buf180, buf182, buf189, buf191, buf201, buf202, buf204, buf211, buf212, buf214, buf221, buf223, buf225, buf232, buf233, buf235, buf242, buf243, buf245, buf252, buf255, buf257, buf264, buf265, buf267, buf274, buf275, buf277, buf284, buf285, buf287, buf294, buf295, buf297, buf304, buf305, buf307, buf314, buf315, buf317, buf324, buf328, buf330, buf337, buf338, buf340, buf347, buf348, buf350, buf357, buf358, buf360, buf367, buf368, buf370, buf377, buf378, buf380, buf387, buf388, buf390, buf397, buf400, buf402, buf409, buf410, buf412, buf419, buf420, buf422, buf429, buf430, buf432, buf439, buf440, buf442, buf449, buf450, buf452, buf459, buf460, buf462, buf469, buf475, buf477, buf484, buf485, buf486, buf487, buf489, buf496, buf498, buf505, buf506, buf508, buf515, buf516, buf518, buf525, buf527, buf529, buf536, buf537, buf539, buf546, buf547, buf549, buf556, buf559, buf561, buf568, buf569, buf571, buf578, buf579, buf581, buf588, buf589, buf591, buf598, buf599, buf601, buf608, buf609, buf611, buf618, buf619, buf621, buf628, buf632, buf634, buf641, buf642, buf644, buf651, buf652, buf654, buf661, buf662, buf664, buf671, buf672, buf674, buf681, buf682, buf684, buf691, buf692, buf694, buf701, buf704, buf706, buf713, buf714, buf716, buf723, buf724, buf726, buf733, buf734, buf736, buf743, buf744, buf746, buf753, buf754, buf756, buf763, buf764, buf766, buf773, buf778, buf780, buf787, buf788, buf790, buf797, buf798, buf800, buf807, buf808, buf810, buf817, buf818, buf820, buf827, buf828, buf830, buf837, buf838, buf840, buf847, buf850, buf852, buf859, buf860, buf862, buf869, buf870, buf872, buf879, buf880, buf882, buf889, buf890, buf892, buf899, buf900, buf902, buf909, buf910, buf912, buf919, buf923, buf925, buf932, buf933, buf935, buf942, buf943, buf945, buf952, buf953, buf955, buf962, buf963, buf965, buf972, buf973, buf975, buf982, buf983, buf985, buf992, buf995, buf997, buf1004, buf1005, buf1007, buf1014, buf1015, buf1017, buf1024, buf1025, buf1027, buf1034, buf1035, buf1037, buf1044, buf1045, buf1047, buf1054, buf1055, buf1057, buf1064, buf1071, buf1073, buf1080, buf1081, buf1082, buf1083, buf1085, buf1092, buf1094, buf1101, buf1102, buf1104, buf1111, buf1112, buf1114, buf1121, buf1123, buf1125, buf1132, buf1133, buf1135, buf1142, buf1143, buf1145, buf1152, buf1156, buf1158, buf1165, buf1168, buf1171, reinterpret_tensor(buf1162, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf1172, reinterpret_tensor(buf1149, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf1139, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf1129, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf1118, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf1108, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf1098, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf1089, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf1077, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1173, reinterpret_tensor(buf1061, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf1051, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf1041, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf1031, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf1021, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf1011, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf1001, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1174, reinterpret_tensor(buf989, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf979, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf969, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf959, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf949, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf939, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf929, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1175, reinterpret_tensor(buf916, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf906, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf896, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf886, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf876, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf866, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf856, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1176, reinterpret_tensor(buf844, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf834, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf824, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf814, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf804, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf794, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf784, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1177, reinterpret_tensor(buf770, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf760, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf750, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf740, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf730, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf720, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf710, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1178, reinterpret_tensor(buf698, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf688, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf678, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf668, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf658, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf648, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf638, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1179, reinterpret_tensor(buf625, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf615, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf605, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf595, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf585, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf575, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf565, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf1180, reinterpret_tensor(buf553, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf543, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf533, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf522, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf512, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf502, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf493, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf481, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf1181, reinterpret_tensor(buf466, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf456, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf446, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf436, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf426, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf416, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf406, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf1182, reinterpret_tensor(buf394, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf384, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf374, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf364, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf354, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf344, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf334, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf1183, reinterpret_tensor(buf321, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf311, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf301, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf291, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf281, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf271, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf261, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf1184, reinterpret_tensor(buf249, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf239, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf229, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf218, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf208, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf198, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf186, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf174, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf1185, reinterpret_tensor(buf159, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf146, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf133, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf119, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf106, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf93, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf81, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf66, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf53, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf40, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((512, 2816, 1, 1), (2816, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1024, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1000, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_321 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_324 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_327 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_330 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_333 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_336 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_339 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_342 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_345 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_348 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_351 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_354 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_357 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_360 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_363 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_366 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_369 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_372 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_375 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_378 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_381 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_384 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_387 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_390 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_393 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_396 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_399 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_402 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_405 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_408 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_411 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_414 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_417 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_420 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_423 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_426 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_429 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_432 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_435 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_438 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_441 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_444 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_447 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_450 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_453 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_456 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_459 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_462 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_465 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_468 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_471 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_474 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_477 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_480 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_483 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_486 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_489 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_492 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_495 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_498 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_501 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_504 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_507 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_510 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_513 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_516 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_519 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_522 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_525 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_528 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_531 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_534 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_537 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_540 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_543 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_546 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_549 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_552 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_555 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_558 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_561 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_564 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_567 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_570 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_573 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_576 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_579 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_582 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_585 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_588 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_591 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_594 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_597 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_600 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_603 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_606 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_609 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_612 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_615 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_618 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_621 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_624 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_627 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_630 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_633 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dla102', benchmark_compiled_module)
