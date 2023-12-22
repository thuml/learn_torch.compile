
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


# kernel path: /tmp/torchinductor_youkaichao/oj/coj2qz6zhg6cwzyzlbxtlutq6rutusz7lgvgusnri2egzt62fqwq.py
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
    size_hints=[512, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (48*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/27/c27ujougsgmrutlg7hg4pengbyk5r5sfwmipn46m6r243xy46oil.py
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
    size_hints=[32768, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (512*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2fu2eqmohsq34a6herppdulrxwihrckwwx2obg6irzmupqhskg.py
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
    size_hints=[131072, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (1024*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coac4tg6mjcbin33bf7t3phx36ls5sqzqaousfgh3y67imfd6cnj.py
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
    size_hints=[524288, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (2048*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zy77nj6tcqsz3llgsoiffa3sfxbhxhev736n5gyv2gppyglta3.py
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
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ps/cpssxx4znruwidxv7nzubk4v5ngzrl76worqttn4c4os3xxsn6ny.py
# Source Nodes: [x_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_1 => add, clone, rsqrt, var_mean
triton_red_fused_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(out_ptr1 + (x3), tmp5, xmask)
    x4 = xindex % 56
    x5 = (xindex // 56) % 56
    tmp7 = 128.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = tl.math.rsqrt(tmp10)
    tmp12 = tmp11 / tmp7
    tl.store(out_ptr2 + (x5 + (56*x4) + (3136*x1)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yt/cyt4hleflbyzsyluf5urc3oxribfl2gnjolsxfeyqrfyapmsyl5x.py
# Source Nodes: [x_1], Original ATen: [aten.native_layer_norm]
# x_1 => add, clone, mul, rsqrt, sub, var_mean
triton_poi_fused_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 128
    x3 = (xindex // 128)
    y0 = yindex % 56
    y1 = (yindex // 56)
    y4 = yindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (56*y0) + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 + (56*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3 + (56*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 128.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (y0 + (56*x5) + (401408*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nl/cnlucnqs6fm3fjiopgytsksrzdkz7ngurk2o7ok4cbpkrl7tw6rh.py
# Source Nodes: [x_1, x_3], Original ATen: [aten.native_layer_norm, aten.permute]
# x_1 => add_1, mul_1
# x_3 => permute_1
triton_poi_fused_native_layer_norm_permute_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_permute_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 56
    x3 = (xindex // 56)
    y0 = yindex % 128
    y1 = (yindex // 128)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (56*y0) + (7168*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y0 + (128*x4) + (401408*y1)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clbcpreeyoddcusvk76zfzyvitpiofxyopr2kpmutmv7ekllz2bn.py
# Source Nodes: [x_8, x_9], Original ATen: [aten.native_layer_norm, aten.view]
# x_8 => add_3, mul_3
# x_9 => view
triton_poi_fused_native_layer_norm_view_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((56*x0) + (7168*(x1 % 56)) + (401408*(x1 // 3136)) + ((x1 // 56) % 56)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dpesl47ddrdwq65ivarbuoxjtqwak652px2oy4vm5kg52ratxj.py
# Source Nodes: [x_10, x_13], Original ATen: [aten.gelu, aten.view]
# x_10 => add_4, erf, mul_4, mul_5, mul_6
# x_13 => view_2
triton_poi_fused_gelu_view_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
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


# kernel path: /tmp/torchinductor_youkaichao/52/c527jfwgmjckchltxro6alq2r7qm6igcmsvsl5vsvztvwj43jtcf.py
# Source Nodes: [shortcut_1, x_17], Original ATen: [aten.add, aten.mul]
# shortcut_1 => add_5
# x_17 => mul_7
triton_poi_fused_add_mul_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kf/ckf3wvjntyfszbcwbuuri5uv23hxujs7ccugwemzw4su53bcblvj.py
# Source Nodes: [x_49], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_49 => add_14, rsqrt_4, var_mean_4
triton_per_fused_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 56
    x3 = (xindex // 56) % 56
    x4 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 128.0
    tmp22 = tmp20 / tmp21
    tmp23 = 1e-06
    tmp24 = tmp22 + tmp23
    tmp25 = tl.math.rsqrt(tmp24)
    tmp26 = tmp25 / tmp21
    tl.store(out_ptr2 + (x3 + (56*x2) + (3136*x4)), tmp26, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bd/cbdnhygn3ojqakvkqyzwwjwrojffnwdjcl3jwdspygm35aqm3x2e.py
# Source Nodes: [x_49], Original ATen: [aten.native_layer_norm]
# x_49 => add_14, mul_20, rsqrt_4, sub_4, var_mean_4
triton_poi_fused_native_layer_norm_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x2 = xindex % 128
    x3 = (xindex // 128)
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3 + (56*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x3 + (56*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 128.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (56*x5) + (401408*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kg/ckgvswdsbtvat2njz23zjflycgxuh7lf4rmgwphi3k57z47qsuvx.py
# Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
# shortcut_3 => convolution_4
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (256*x2) + (200704*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eb/cebmds7jx5hjdmwojpvgezkarqvjfombthnbgbnhqc7x2ndlbatv.py
# Source Nodes: [x_55], Original ATen: [aten.native_layer_norm]
# x_55 => clone_10, var_mean_5
triton_red_fused_native_layer_norm_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 2
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r3) + (100352*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhrwowbqanvz2dde7syuldew2o3l6ejtxblsnklbtbfcgoies57.py
# Source Nodes: [x_55], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_55 => add_16, clone_10, rsqrt_5, var_mean_5
triton_per_fused_native_layer_norm_native_layer_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    x4 = xindex % 28
    x5 = (xindex // 28) % 28
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (1568*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (784*r2) + (1568*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (784*r2) + (1568*x1)), rmask & xmask, other=0.0)
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
    tmp16 = 256.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x5 + (28*x4) + (784*x1)), tmp21, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zu/czuj6koklssolaijnezplkci4xq4k3aeg72c5vdgbtl2ym4joz55.py
# Source Nodes: [x_55], Original ATen: [aten.native_layer_norm]
# x_55 => add_16, clone_10, mul_22, rsqrt_5, sub_5, var_mean_5
triton_poi_fused_native_layer_norm_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 256
    x3 = (xindex // 256)
    y0 = yindex % 28
    y1 = (yindex // 28)
    y4 = yindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (28*y0) + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 + (28*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3 + (28*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (y0 + (28*x5) + (200704*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbzouwgijwkom2tk4eykbfrszy3t2crawvtowwc443xiuprawur.py
# Source Nodes: [x_55, x_56], Original ATen: [aten.native_layer_norm, aten.view]
# x_55 => add_17, mul_23
# x_56 => view_15
triton_poi_fused_native_layer_norm_view_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((28*x0) + (7168*(x1 % 28)) + (200704*(x1 // 784)) + ((x1 // 28) % 28)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfrfvi3zaw6v6pl3q6i4k6szje5vze6ei375tbgonrwxqsooanqh.py
# Source Nodes: [x_57, x_60], Original ATen: [aten.gelu, aten.view]
# x_57 => add_18, erf_3, mul_24, mul_25, mul_26
# x_60 => view_17
triton_poi_fused_gelu_view_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_youkaichao/yv/cyvh25rlb7tbhf6nlevrkdagiaoedwtzevbtf7b7ydd7el3mdykv.py
# Source Nodes: [shortcut_4, x_64], Original ATen: [aten.add, aten.mul]
# shortcut_4 => add_19
# x_64 => mul_27
triton_poi_fused_add_mul_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qz/cqzqxsxhfka5lai3sjb4p3ao4fsmd43fr7hhbcesz4dtvako4kx4.py
# Source Nodes: [x_96], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_96 => add_28, rsqrt_8, var_mean_8
triton_per_fused_native_layer_norm_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 6272
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
    x2 = xindex % 28
    x3 = (xindex // 28) % 28
    x4 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 256, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 256.0
    tmp22 = tmp20 / tmp21
    tmp23 = 1e-06
    tmp24 = tmp22 + tmp23
    tmp25 = tl.math.rsqrt(tmp24)
    tmp26 = tmp25 / tmp21
    tl.store(out_ptr2 + (x3 + (28*x2) + (784*x4)), tmp26, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k7/ck7wgubuouyt2xr6bxmhxrwfaowmugmt4p6vzt4shczatg2yp3id.py
# Source Nodes: [x_96], Original ATen: [aten.native_layer_norm]
# x_96 => add_28, mul_40, rsqrt_8, sub_8, var_mean_8
triton_poi_fused_native_layer_norm_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x2 = xindex % 256
    x3 = (xindex // 256)
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3 + (28*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x3 + (28*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 256.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (28*x5) + (200704*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xtvekio2idndnjpgqygef2c3zzufjnkas5fczz3kolclqsuipj.py
# Source Nodes: [x_96, x_97], Original ATen: [aten.native_layer_norm, aten.permute]
# x_96 => add_29, mul_41
# x_97 => permute_29
triton_poi_fused_native_layer_norm_permute_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_permute_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 28
    x3 = (xindex // 28)
    y0 = yindex % 256
    y1 = (yindex // 256)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (28*y0) + (7168*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y0 + (256*x4) + (200704*y1)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6r/c6rp6rktep34kqvb44tequueek77fsczoy7mlnie2y35iz2jt4n3.py
# Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
# shortcut_6 => convolution_8
triton_poi_fused_convolution_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (512*x2) + (100352*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ks/cksruahipcpk62zhhuxyckzoze73rixgm3f4oguqdpbxtwubft2b.py
# Source Nodes: [x_102], Original ATen: [aten.native_layer_norm]
# x_102 => clone_19, var_mean_9
triton_red_fused_native_layer_norm_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 4
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2plbpszjksfvpjxgpiv6jwlqc7rkkivy7cd3nnjwdujiqci7qe.py
# Source Nodes: [x_102], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_102 => add_30, clone_19, rsqrt_9, var_mean_9
triton_per_fused_native_layer_norm_native_layer_norm_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    x4 = xindex % 14
    x5 = (xindex // 14) % 14
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (784*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (784*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (784*x1)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x5 + (14*x4) + (196*x1)), tmp21, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lsv3gvk7qdteeyvbsjdlzyg5nrb2xgmeovin7lqoswadqoidfw.py
# Source Nodes: [x_102], Original ATen: [aten.native_layer_norm]
# x_102 => add_30, clone_19, mul_42, rsqrt_9, sub_9, var_mean_9
triton_poi_fused_native_layer_norm_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = (yindex // 196)
    y4 = yindex % 196
    y5 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14) % 14
    tmp0 = tl.load(in_ptr0 + (y4 + (196*x3) + (100352*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y5), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y5), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 512.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (y1 + (14*x3) + (7168*y0) + (100352*y2)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e2/ce23u3rqrz3w3aohxxpxksj44rwfwnlyg2wznnpnm2ysnf276eq6.py
# Source Nodes: [x_102, x_103], Original ATen: [aten.native_layer_norm, aten.view]
# x_102 => add_31, mul_43
# x_103 => view_30
triton_poi_fused_native_layer_norm_view_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((14*x0) + (7168*(x1 % 14)) + (100352*(x1 // 196)) + ((x1 // 14) % 14)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozx7aqevn3czlcxvkgxjsi44psm2zfl2bxmm7g3hncisew6f4wl.py
# Source Nodes: [x_104, x_107], Original ATen: [aten.gelu, aten.view]
# x_104 => add_32, erf_6, mul_44, mul_45, mul_46
# x_107 => view_32
triton_poi_fused_gelu_view_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbjqtx3x34odf2dmtjc3jyga7xvcyktjtkkahwqlvvmaqrbd65x.py
# Source Nodes: [shortcut_7, x_111], Original ATen: [aten.add, aten.mul]
# shortcut_7 => add_33
# x_111 => mul_47
triton_poi_fused_add_mul_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qm/cqmdwichl2okv55232rynwx5mz6rkkfjswt5g36g3i6b263kteq3.py
# Source Nodes: [x_479], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_479 => add_138, rsqrt_36, var_mean_36
triton_per_fused_native_layer_norm_native_layer_norm_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x2 = xindex % 14
    x3 = (xindex // 14) % 14
    x4 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 512.0
    tmp22 = tmp20 / tmp21
    tmp23 = 1e-06
    tmp24 = tmp22 + tmp23
    tmp25 = tl.math.rsqrt(tmp24)
    tmp26 = tmp25 / tmp21
    tl.store(out_ptr2 + (x3 + (14*x2) + (196*x4)), tmp26, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlkq4ljtf5pe5wumpio3254tvbloj3dz2v3e6yyaxtzrk56iz2c.py
# Source Nodes: [x_479], Original ATen: [aten.native_layer_norm]
# x_479 => add_138, mul_204, rsqrt_36, sub_36, var_mean_36
triton_poi_fused_native_layer_norm_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x2 = xindex % 512
    x3 = (xindex // 512)
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3 + (14*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x3 + (14*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 512.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (14*x5) + (100352*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lm/clmcfb2oapi6i4dobu5tp5vmz6l2jb2dmsxagj2v5rntorzbq7cj.py
# Source Nodes: [x_479, x_480], Original ATen: [aten.native_layer_norm, aten.permute]
# x_479 => add_139, mul_205
# x_480 => permute_139
triton_poi_fused_native_layer_norm_permute_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_permute_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 14
    x3 = (xindex // 14)
    y0 = yindex % 512
    y1 = (yindex // 512)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (14*y0) + (7168*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y0 + (512*x4) + (100352*y1)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l6/cl6ogj5p4enoh46m4zwxh4w266whfufulyg6p4n4ur53xilxgr3o.py
# Source Nodes: [shortcut_33], Original ATen: [aten.convolution]
# shortcut_33 => convolution_36
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (1024*x2) + (50176*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4vaq6ytynl4jjdsozfv4mi7e2imwz6hnyybm6jholagbncntg4.py
# Source Nodes: [x_485], Original ATen: [aten.native_layer_norm]
# x_485 => clone_100, var_mean_37
triton_red_fused_native_layer_norm_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 8
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (49*r3) + (6272*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5j/c5jtjz4lgk5bugqlzi3jyu5qe3yj273qlo4wn7zmevjzrcybf4bs.py
# Source Nodes: [x_485], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_485 => add_140, clone_100, rsqrt_37, var_mean_37
triton_per_fused_native_layer_norm_native_layer_norm_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    x4 = xindex % 7
    x5 = (xindex // 7) % 7
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (392*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (49*r2) + (392*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (49*r2) + (392*x1)), rmask & xmask, other=0.0)
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
    tmp16 = 1024.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x5 + (7*x4) + (49*x1)), tmp21, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xfctywzybc6nc6w4kntu45inzxonm3gvzywpnjgs2b6noxrb5z.py
# Source Nodes: [x_485], Original ATen: [aten.native_layer_norm]
# x_485 => add_140, clone_100, mul_206, rsqrt_37, sub_37, var_mean_37
triton_poi_fused_native_layer_norm_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 56
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 1024
    x3 = (xindex // 1024)
    y0 = yindex % 7
    y1 = (yindex // 7)
    y4 = yindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (7*y0) + (49*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 + (7*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3 + (7*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1024.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (y0 + (7*x5) + (50176*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpj4ju773rf6h7juhv2o24heoep7ne7q3m3dpmiy5l6clwow5my.py
# Source Nodes: [x_485, x_486], Original ATen: [aten.native_layer_norm, aten.view]
# x_485 => add_141, mul_207
# x_486 => view_165
triton_poi_fused_native_layer_norm_view_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((7*x0) + (7168*(x1 % 7)) + (50176*(x1 // 49)) + ((x1 // 7) % 7)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lk/clk4s7fwbsphja7bt3ou2xoydpxyuvr562agip3tgperdbyrvkzq.py
# Source Nodes: [x_487, x_490], Original ATen: [aten.gelu, aten.view]
# x_487 => add_142, erf_33, mul_208, mul_209, mul_210
# x_490 => view_167
triton_poi_fused_gelu_view_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6ykfd3fqlhmpnrwe2qbr3xvs56uufrfn34xyald65feu52cfkr.py
# Source Nodes: [shortcut_34, x_494], Original ATen: [aten.add, aten.mul]
# shortcut_34 => add_143
# x_494 => mul_211
triton_poi_fused_add_mul_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63a3xsfgmnuvvui25bdqlkfazgcdynyj3g5lvumzn7cqcv5mo4b.py
# Source Nodes: [x_522, x_525, x_528], Original ATen: [aten.add, aten.mean, aten.mul]
# x_522 => mul_223
# x_525 => add_151
# x_528 => mean
triton_per_fused_add_mean_mul_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_40', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (1024*r2) + (50176*x1)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6ptdprju6r7gmlgjx2vvo6spyd5aplw3eeyk4y5jqxzvt4pksw.py
# Source Nodes: [x_532, x_535], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# x_532 => add_152, mul_224, rsqrt_40, sub_40, var_mean_40
# x_535 => view_180
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 1024.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp22 / tmp18
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345 = args
    args.clear()
    assert_size_stride(primals_1, (128, ), (1, ))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, ), (1, ))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_47, (512, ), (1, ))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_59, (512, ), (1, ))
    assert_size_stride(primals_60, (512, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (512, ), (1, ))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (512, ), (1, ))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_77, (512, ), (1, ))
    assert_size_stride(primals_78, (512, ), (1, ))
    assert_size_stride(primals_79, (512, ), (1, ))
    assert_size_stride(primals_80, (512, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_92, (512, ), (1, ))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_101, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_108, (1024, ), (1, ))
    assert_size_stride(primals_109, (1024, ), (1, ))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_112, (1024, ), (1, ))
    assert_size_stride(primals_113, (1024, ), (1, ))
    assert_size_stride(primals_114, (1024, ), (1, ))
    assert_size_stride(primals_115, (1024, ), (1, ))
    assert_size_stride(primals_116, (1024, ), (1, ))
    assert_size_stride(primals_117, (1024, ), (1, ))
    assert_size_stride(primals_118, (1024, ), (1, ))
    assert_size_stride(primals_119, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (512, 128), (128, 1))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_125, (128, 512), (512, 1))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (512, 128), (128, 1))
    assert_size_stride(primals_130, (512, ), (1, ))
    assert_size_stride(primals_131, (128, 512), (512, 1))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (512, 128), (128, 1))
    assert_size_stride(primals_136, (512, ), (1, ))
    assert_size_stride(primals_137, (128, 512), (512, 1))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (256, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_141, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_142, (256, ), (1, ))
    assert_size_stride(primals_143, (1024, 256), (256, 1))
    assert_size_stride(primals_144, (1024, ), (1, ))
    assert_size_stride(primals_145, (256, 1024), (1024, 1))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_147, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_148, (256, ), (1, ))
    assert_size_stride(primals_149, (1024, 256), (256, 1))
    assert_size_stride(primals_150, (1024, ), (1, ))
    assert_size_stride(primals_151, (256, 1024), (1024, 1))
    assert_size_stride(primals_152, (256, ), (1, ))
    assert_size_stride(primals_153, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_154, (256, ), (1, ))
    assert_size_stride(primals_155, (1024, 256), (256, 1))
    assert_size_stride(primals_156, (1024, ), (1, ))
    assert_size_stride(primals_157, (256, 1024), (1024, 1))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_159, (512, 256, 2, 2), (1024, 4, 2, 1))
    assert_size_stride(primals_160, (512, ), (1, ))
    assert_size_stride(primals_161, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_162, (512, ), (1, ))
    assert_size_stride(primals_163, (2048, 512), (512, 1))
    assert_size_stride(primals_164, (2048, ), (1, ))
    assert_size_stride(primals_165, (512, 2048), (2048, 1))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_168, (512, ), (1, ))
    assert_size_stride(primals_169, (2048, 512), (512, 1))
    assert_size_stride(primals_170, (2048, ), (1, ))
    assert_size_stride(primals_171, (512, 2048), (2048, 1))
    assert_size_stride(primals_172, (512, ), (1, ))
    assert_size_stride(primals_173, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_174, (512, ), (1, ))
    assert_size_stride(primals_175, (2048, 512), (512, 1))
    assert_size_stride(primals_176, (2048, ), (1, ))
    assert_size_stride(primals_177, (512, 2048), (2048, 1))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (2048, 512), (512, 1))
    assert_size_stride(primals_182, (2048, ), (1, ))
    assert_size_stride(primals_183, (512, 2048), (2048, 1))
    assert_size_stride(primals_184, (512, ), (1, ))
    assert_size_stride(primals_185, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_186, (512, ), (1, ))
    assert_size_stride(primals_187, (2048, 512), (512, 1))
    assert_size_stride(primals_188, (2048, ), (1, ))
    assert_size_stride(primals_189, (512, 2048), (2048, 1))
    assert_size_stride(primals_190, (512, ), (1, ))
    assert_size_stride(primals_191, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_192, (512, ), (1, ))
    assert_size_stride(primals_193, (2048, 512), (512, 1))
    assert_size_stride(primals_194, (2048, ), (1, ))
    assert_size_stride(primals_195, (512, 2048), (2048, 1))
    assert_size_stride(primals_196, (512, ), (1, ))
    assert_size_stride(primals_197, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_198, (512, ), (1, ))
    assert_size_stride(primals_199, (2048, 512), (512, 1))
    assert_size_stride(primals_200, (2048, ), (1, ))
    assert_size_stride(primals_201, (512, 2048), (2048, 1))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_203, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (2048, 512), (512, 1))
    assert_size_stride(primals_206, (2048, ), (1, ))
    assert_size_stride(primals_207, (512, 2048), (2048, 1))
    assert_size_stride(primals_208, (512, ), (1, ))
    assert_size_stride(primals_209, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_210, (512, ), (1, ))
    assert_size_stride(primals_211, (2048, 512), (512, 1))
    assert_size_stride(primals_212, (2048, ), (1, ))
    assert_size_stride(primals_213, (512, 2048), (2048, 1))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_216, (512, ), (1, ))
    assert_size_stride(primals_217, (2048, 512), (512, 1))
    assert_size_stride(primals_218, (2048, ), (1, ))
    assert_size_stride(primals_219, (512, 2048), (2048, 1))
    assert_size_stride(primals_220, (512, ), (1, ))
    assert_size_stride(primals_221, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (2048, 512), (512, 1))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_225, (512, 2048), (2048, 1))
    assert_size_stride(primals_226, (512, ), (1, ))
    assert_size_stride(primals_227, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_228, (512, ), (1, ))
    assert_size_stride(primals_229, (2048, 512), (512, 1))
    assert_size_stride(primals_230, (2048, ), (1, ))
    assert_size_stride(primals_231, (512, 2048), (2048, 1))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_233, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (2048, 512), (512, 1))
    assert_size_stride(primals_236, (2048, ), (1, ))
    assert_size_stride(primals_237, (512, 2048), (2048, 1))
    assert_size_stride(primals_238, (512, ), (1, ))
    assert_size_stride(primals_239, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_240, (512, ), (1, ))
    assert_size_stride(primals_241, (2048, 512), (512, 1))
    assert_size_stride(primals_242, (2048, ), (1, ))
    assert_size_stride(primals_243, (512, 2048), (2048, 1))
    assert_size_stride(primals_244, (512, ), (1, ))
    assert_size_stride(primals_245, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (2048, 512), (512, 1))
    assert_size_stride(primals_248, (2048, ), (1, ))
    assert_size_stride(primals_249, (512, 2048), (2048, 1))
    assert_size_stride(primals_250, (512, ), (1, ))
    assert_size_stride(primals_251, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_252, (512, ), (1, ))
    assert_size_stride(primals_253, (2048, 512), (512, 1))
    assert_size_stride(primals_254, (2048, ), (1, ))
    assert_size_stride(primals_255, (512, 2048), (2048, 1))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_257, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (2048, 512), (512, 1))
    assert_size_stride(primals_260, (2048, ), (1, ))
    assert_size_stride(primals_261, (512, 2048), (2048, 1))
    assert_size_stride(primals_262, (512, ), (1, ))
    assert_size_stride(primals_263, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (2048, 512), (512, 1))
    assert_size_stride(primals_266, (2048, ), (1, ))
    assert_size_stride(primals_267, (512, 2048), (2048, 1))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_269, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_271, (2048, 512), (512, 1))
    assert_size_stride(primals_272, (2048, ), (1, ))
    assert_size_stride(primals_273, (512, 2048), (2048, 1))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_275, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_276, (512, ), (1, ))
    assert_size_stride(primals_277, (2048, 512), (512, 1))
    assert_size_stride(primals_278, (2048, ), (1, ))
    assert_size_stride(primals_279, (512, 2048), (2048, 1))
    assert_size_stride(primals_280, (512, ), (1, ))
    assert_size_stride(primals_281, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_282, (512, ), (1, ))
    assert_size_stride(primals_283, (2048, 512), (512, 1))
    assert_size_stride(primals_284, (2048, ), (1, ))
    assert_size_stride(primals_285, (512, 2048), (2048, 1))
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_287, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_289, (2048, 512), (512, 1))
    assert_size_stride(primals_290, (2048, ), (1, ))
    assert_size_stride(primals_291, (512, 2048), (2048, 1))
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (2048, 512), (512, 1))
    assert_size_stride(primals_296, (2048, ), (1, ))
    assert_size_stride(primals_297, (512, 2048), (2048, 1))
    assert_size_stride(primals_298, (512, ), (1, ))
    assert_size_stride(primals_299, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_300, (512, ), (1, ))
    assert_size_stride(primals_301, (2048, 512), (512, 1))
    assert_size_stride(primals_302, (2048, ), (1, ))
    assert_size_stride(primals_303, (512, 2048), (2048, 1))
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (2048, 512), (512, 1))
    assert_size_stride(primals_308, (2048, ), (1, ))
    assert_size_stride(primals_309, (512, 2048), (2048, 1))
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_311, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_313, (2048, 512), (512, 1))
    assert_size_stride(primals_314, (2048, ), (1, ))
    assert_size_stride(primals_315, (512, 2048), (2048, 1))
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_318, (512, ), (1, ))
    assert_size_stride(primals_319, (2048, 512), (512, 1))
    assert_size_stride(primals_320, (2048, ), (1, ))
    assert_size_stride(primals_321, (512, 2048), (2048, 1))
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (1024, 512, 2, 2), (2048, 4, 2, 1))
    assert_size_stride(primals_324, (1024, ), (1, ))
    assert_size_stride(primals_325, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_326, (1024, ), (1, ))
    assert_size_stride(primals_327, (4096, 1024), (1024, 1))
    assert_size_stride(primals_328, (4096, ), (1, ))
    assert_size_stride(primals_329, (1024, 4096), (4096, 1))
    assert_size_stride(primals_330, (1024, ), (1, ))
    assert_size_stride(primals_331, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_332, (1024, ), (1, ))
    assert_size_stride(primals_333, (4096, 1024), (1024, 1))
    assert_size_stride(primals_334, (4096, ), (1, ))
    assert_size_stride(primals_335, (1024, 4096), (4096, 1))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_338, (1024, ), (1, ))
    assert_size_stride(primals_339, (4096, 1024), (1024, 1))
    assert_size_stride(primals_340, (4096, ), (1, ))
    assert_size_stride(primals_341, (1024, 4096), (4096, 1))
    assert_size_stride(primals_342, (1024, ), (1, ))
    assert_size_stride(primals_343, (1000, 1024), (1024, 1))
    assert_size_stride(primals_344, (1000, ), (1, ))
    assert_size_stride(primals_345, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((128, 3, 4, 4), (48, 1, 12, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_119, buf0, 384, 16, grid=grid(384, 16), stream=stream0)
        del primals_119
        buf1 = empty_strided((256, 128, 2, 2), (512, 1, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_139, buf1, 32768, 4, grid=grid(32768, 4), stream=stream0)
        del primals_139
        buf2 = empty_strided((512, 256, 2, 2), (1024, 1, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_159, buf2, 131072, 4, grid=grid(131072, 4), stream=stream0)
        del primals_159
        buf3 = empty_strided((1024, 512, 2, 2), (2048, 1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_323, buf3, 524288, 4, grid=grid(524288, 4), stream=stream0)
        del primals_323
        buf4 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_345, buf4, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_345
        # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, buf0, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf6 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cuda', dtype=torch.float32)
        buf535 = empty_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_5.run(buf5, primals_120, buf6, buf7, buf535, 25088, 128, grid=grid(25088), stream=stream0)
        buf9 = empty_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_6.run(buf5, primals_120, buf6, buf7, buf9, 448, 7168, grid=grid(448, 7168), stream=stream0)
        del primals_120
        buf10 = reinterpret_tensor(buf5, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf5  # reuse
        # Source Nodes: [x_1, x_3], Original ATen: [aten.native_layer_norm, aten.permute]
        triton_poi_fused_native_layer_norm_permute_7.run(buf9, primals_1, primals_2, buf10, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_121, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf11, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf12 = buf7; del buf7  # reuse
        buf13 = buf6; del buf6  # reuse
        buf534 = empty_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_5.run(buf11, primals_122, buf12, buf13, buf534, 25088, 128, grid=grid(25088), stream=stream0)
        buf15 = empty_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_6.run(buf11, primals_122, buf12, buf13, buf15, 448, 7168, grid=grid(448, 7168), stream=stream0)
        del primals_122
        buf16 = reinterpret_tensor(buf11, (25088, 128), (128, 1), 0); del buf11  # reuse
        # Source Nodes: [x_8, x_9], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_8.run(buf15, primals_3, primals_4, buf16, 3211264, grid=grid(3211264), stream=stream0)
        del primals_4
        buf17 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_124, buf16, reinterpret_tensor(primals_123, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf17)
        del primals_124
        buf18 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10, x_13], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_9.run(buf17, buf18, 12845056, grid=grid(12845056), stream=stream0)
        buf19 = empty((25088, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_126, buf18, reinterpret_tensor(primals_125, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf19)
        del primals_126
        buf20 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_1, x_17], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_10.run(buf19, primals_5, buf10, buf20, 3211264, grid=grid(3211264), stream=stream0)
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_127, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf21, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf22 = buf13; del buf13  # reuse
        buf23 = buf12; del buf12  # reuse
        buf533 = empty_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_5.run(buf21, primals_128, buf22, buf23, buf533, 25088, 128, grid=grid(25088), stream=stream0)
        buf25 = empty_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_6.run(buf21, primals_128, buf22, buf23, buf25, 448, 7168, grid=grid(448, 7168), stream=stream0)
        del primals_128
        buf26 = reinterpret_tensor(buf21, (25088, 128), (128, 1), 0); del buf21  # reuse
        # Source Nodes: [x_22, x_23], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_8.run(buf25, primals_6, primals_7, buf26, 3211264, grid=grid(3211264), stream=stream0)
        del primals_7
        buf27 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_130, buf26, reinterpret_tensor(primals_129, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf27)
        del primals_130
        buf28 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24, x_27], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_9.run(buf27, buf28, 12845056, grid=grid(12845056), stream=stream0)
        buf29 = empty((25088, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_132, buf28, reinterpret_tensor(primals_131, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf29)
        del primals_132
        buf30 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_2, x_31], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_10.run(buf29, primals_8, buf20, buf30, 3211264, grid=grid(3211264), stream=stream0)
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_133, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf31, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf32 = buf23; del buf23  # reuse
        buf33 = buf22; del buf22  # reuse
        buf532 = empty_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_5.run(buf31, primals_134, buf32, buf33, buf532, 25088, 128, grid=grid(25088), stream=stream0)
        buf35 = empty_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_6.run(buf31, primals_134, buf32, buf33, buf35, 448, 7168, grid=grid(448, 7168), stream=stream0)
        del primals_134
        buf36 = reinterpret_tensor(buf31, (25088, 128), (128, 1), 0); del buf31  # reuse
        # Source Nodes: [x_36, x_37], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_8.run(buf35, primals_9, primals_10, buf36, 3211264, grid=grid(3211264), stream=stream0)
        del primals_10
        buf37 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_136, buf36, reinterpret_tensor(primals_135, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf37)
        del primals_136
        buf38 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38, x_41], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_9.run(buf37, buf38, 12845056, grid=grid(12845056), stream=stream0)
        buf39 = empty((25088, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_138, buf38, reinterpret_tensor(primals_137, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf39)
        del primals_138
        buf40 = buf33; del buf33  # reuse
        buf41 = buf32; del buf32  # reuse
        buf531 = empty_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_11.run(buf39, primals_11, buf30, buf40, buf41, buf531, 25088, 128, grid=grid(25088), stream=stream0)
        buf43 = empty_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_12.run(buf39, primals_11, buf30, buf40, buf41, buf43, 448, 7168, grid=grid(448, 7168), stream=stream0)
        del buf40
        del buf41
        buf44 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49, x_50], Original ATen: [aten.native_layer_norm, aten.permute]
        triton_poi_fused_native_layer_norm_permute_7.run(buf43, primals_12, primals_13, buf44, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del primals_13
        # Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf46 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf45, primals_140, buf46, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del primals_140
        # Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_141, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf47, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf48 = empty_strided((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), device='cuda', dtype=torch.float32)
        buf50 = empty_strided((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_14.run(buf47, primals_142, buf48, buf49, buf50, 12544, 128, grid=grid(12544), stream=stream0)
        buf51 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        buf52 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        buf530 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_15.run(buf48, buf49, buf50, buf51, buf52, buf530, 6272, 2, grid=grid(6272), stream=stream0)
        buf54 = reinterpret_tensor(buf45, (8, 28, 28, 256), (200704, 1, 7168, 28), 0); del buf45  # reuse
        # Source Nodes: [x_55], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_16.run(buf47, primals_142, buf51, buf52, buf54, 224, 7168, grid=grid(224, 7168), stream=stream0)
        del primals_142
        buf55 = reinterpret_tensor(buf47, (6272, 256), (256, 1), 0); del buf47  # reuse
        # Source Nodes: [x_55, x_56], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_17.run(buf54, primals_14, primals_15, buf55, 1605632, grid=grid(1605632), stream=stream0)
        del primals_15
        buf56 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_144, buf55, reinterpret_tensor(primals_143, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf56)
        del primals_144
        buf57 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57, x_60], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_18.run(buf56, buf57, 6422528, grid=grid(6422528), stream=stream0)
        buf58 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_146, buf57, reinterpret_tensor(primals_145, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf58)
        del primals_146
        buf59 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_4, x_64], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf58, primals_16, buf46, buf59, 1605632, grid=grid(1605632), stream=stream0)
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_147, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf60, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf61 = buf50; del buf50  # reuse
        buf62 = buf49; del buf49  # reuse
        buf63 = buf48; del buf48  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_14.run(buf60, primals_148, buf61, buf62, buf63, 12544, 128, grid=grid(12544), stream=stream0)
        buf64 = buf52; del buf52  # reuse
        buf65 = buf51; del buf51  # reuse
        buf529 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_15.run(buf61, buf62, buf63, buf64, buf65, buf529, 6272, 2, grid=grid(6272), stream=stream0)
        buf67 = empty_strided((8, 28, 28, 256), (200704, 1, 7168, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_16.run(buf60, primals_148, buf64, buf65, buf67, 224, 7168, grid=grid(224, 7168), stream=stream0)
        del primals_148
        buf68 = reinterpret_tensor(buf60, (6272, 256), (256, 1), 0); del buf60  # reuse
        # Source Nodes: [x_69, x_70], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_17.run(buf67, primals_17, primals_18, buf68, 1605632, grid=grid(1605632), stream=stream0)
        del primals_18
        buf69 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_150, buf68, reinterpret_tensor(primals_149, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf69)
        del primals_150
        buf70 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71, x_74], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_18.run(buf69, buf70, 6422528, grid=grid(6422528), stream=stream0)
        buf71 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_152, buf70, reinterpret_tensor(primals_151, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf71)
        del primals_152
        buf72 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5, x_78], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf71, primals_19, buf59, buf72, 1605632, grid=grid(1605632), stream=stream0)
        # Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_153, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf73, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf74 = buf63; del buf63  # reuse
        buf75 = buf62; del buf62  # reuse
        buf76 = buf61; del buf61  # reuse
        # Source Nodes: [x_83], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_14.run(buf73, primals_154, buf74, buf75, buf76, 12544, 128, grid=grid(12544), stream=stream0)
        buf77 = buf65; del buf65  # reuse
        buf78 = buf64; del buf64  # reuse
        buf528 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_15.run(buf74, buf75, buf76, buf77, buf78, buf528, 6272, 2, grid=grid(6272), stream=stream0)
        del buf74
        del buf75
        del buf76
        buf80 = empty_strided((8, 28, 28, 256), (200704, 1, 7168, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_16.run(buf73, primals_154, buf77, buf78, buf80, 224, 7168, grid=grid(224, 7168), stream=stream0)
        del primals_154
        buf81 = reinterpret_tensor(buf73, (6272, 256), (256, 1), 0); del buf73  # reuse
        # Source Nodes: [x_83, x_84], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_17.run(buf80, primals_20, primals_21, buf81, 1605632, grid=grid(1605632), stream=stream0)
        del primals_21
        buf82 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_156, buf81, reinterpret_tensor(primals_155, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf82)
        del primals_156
        buf83 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85, x_88], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_18.run(buf82, buf83, 6422528, grid=grid(6422528), stream=stream0)
        buf84 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_158, buf83, reinterpret_tensor(primals_157, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf84)
        del primals_158
        buf85 = buf78; del buf78  # reuse
        buf86 = buf77; del buf77  # reuse
        buf527 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_20.run(buf84, primals_22, buf72, buf85, buf86, buf527, 6272, 256, grid=grid(6272), stream=stream0)
        buf88 = empty_strided((8, 28, 28, 256), (200704, 1, 7168, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_21.run(buf84, primals_22, buf72, buf85, buf86, buf88, 224, 7168, grid=grid(224, 7168), stream=stream0)
        buf89 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96, x_97], Original ATen: [aten.native_layer_norm, aten.permute]
        triton_poi_fused_native_layer_norm_permute_22.run(buf88, primals_23, primals_24, buf89, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del primals_24
        # Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, buf2, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf91 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf90, primals_160, buf91, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del primals_160
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_161, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf92, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf93 = reinterpret_tensor(buf86, (8, 14, 14, 1, 4), (784, 14, 1, 6272, 196), 0); del buf86  # reuse
        buf94 = reinterpret_tensor(buf85, (8, 14, 14, 1, 4), (784, 14, 1, 6272, 196), 0); del buf85  # reuse
        buf95 = empty_strided((8, 14, 14, 1, 4), (784, 14, 1, 6272, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf92, primals_162, buf93, buf94, buf95, 6272, 128, grid=grid(6272), stream=stream0)
        buf96 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        buf97 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        buf526 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf93, buf94, buf95, buf96, buf97, buf526, 1568, 4, grid=grid(1568), stream=stream0)
        buf99 = reinterpret_tensor(buf90, (8, 14, 14, 512), (100352, 1, 7168, 14), 0); del buf90  # reuse
        # Source Nodes: [x_102], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf92, primals_162, buf96, buf97, buf99, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_162
        buf100 = reinterpret_tensor(buf92, (1568, 512), (512, 1), 0); del buf92  # reuse
        # Source Nodes: [x_102, x_103], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf99, primals_25, primals_26, buf100, 802816, grid=grid(802816), stream=stream0)
        del primals_26
        buf101 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_164, buf100, reinterpret_tensor(primals_163, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf101)
        del primals_164
        buf102 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104, x_107], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf101, buf102, 3211264, grid=grid(3211264), stream=stream0)
        buf103 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_166, buf102, reinterpret_tensor(primals_165, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf103)
        del primals_166
        buf104 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_7, x_111], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf103, primals_27, buf91, buf104, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_113], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_167, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf105, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf106 = buf95; del buf95  # reuse
        buf107 = buf94; del buf94  # reuse
        buf108 = buf93; del buf93  # reuse
        # Source Nodes: [x_116], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf105, primals_168, buf106, buf107, buf108, 6272, 128, grid=grid(6272), stream=stream0)
        buf109 = buf97; del buf97  # reuse
        buf110 = buf96; del buf96  # reuse
        buf525 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf106, buf107, buf108, buf109, buf110, buf525, 1568, 4, grid=grid(1568), stream=stream0)
        buf112 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf105, primals_168, buf109, buf110, buf112, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_168
        buf113 = reinterpret_tensor(buf105, (1568, 512), (512, 1), 0); del buf105  # reuse
        # Source Nodes: [x_116, x_117], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf112, primals_28, primals_29, buf113, 802816, grid=grid(802816), stream=stream0)
        del primals_29
        buf114 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_170, buf113, reinterpret_tensor(primals_169, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf114)
        del primals_170
        buf115 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118, x_121], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf114, buf115, 3211264, grid=grid(3211264), stream=stream0)
        buf116 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_172, buf115, reinterpret_tensor(primals_171, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf116)
        del primals_172
        buf117 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_8, x_125], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf116, primals_30, buf104, buf117, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_173, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf118, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf119 = buf108; del buf108  # reuse
        buf120 = buf107; del buf107  # reuse
        buf121 = buf106; del buf106  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf118, primals_174, buf119, buf120, buf121, 6272, 128, grid=grid(6272), stream=stream0)
        buf122 = buf110; del buf110  # reuse
        buf123 = buf109; del buf109  # reuse
        buf524 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf119, buf120, buf121, buf122, buf123, buf524, 1568, 4, grid=grid(1568), stream=stream0)
        buf125 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf118, primals_174, buf122, buf123, buf125, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_174
        buf126 = reinterpret_tensor(buf118, (1568, 512), (512, 1), 0); del buf118  # reuse
        # Source Nodes: [x_130, x_131], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf125, primals_31, primals_32, buf126, 802816, grid=grid(802816), stream=stream0)
        del primals_32
        buf127 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_131], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_176, buf126, reinterpret_tensor(primals_175, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf127)
        del primals_176
        buf128 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132, x_135], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf127, buf128, 3211264, grid=grid(3211264), stream=stream0)
        buf129 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_178, buf128, reinterpret_tensor(primals_177, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf129)
        del primals_178
        buf130 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_9, x_139], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf129, primals_33, buf117, buf130, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_179, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf131, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf132 = buf121; del buf121  # reuse
        buf133 = buf120; del buf120  # reuse
        buf134 = buf119; del buf119  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf131, primals_180, buf132, buf133, buf134, 6272, 128, grid=grid(6272), stream=stream0)
        buf135 = buf123; del buf123  # reuse
        buf136 = buf122; del buf122  # reuse
        buf523 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf132, buf133, buf134, buf135, buf136, buf523, 1568, 4, grid=grid(1568), stream=stream0)
        buf138 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf131, primals_180, buf135, buf136, buf138, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_180
        buf139 = reinterpret_tensor(buf131, (1568, 512), (512, 1), 0); del buf131  # reuse
        # Source Nodes: [x_144, x_145], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf138, primals_34, primals_35, buf139, 802816, grid=grid(802816), stream=stream0)
        del primals_35
        buf140 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_182, buf139, reinterpret_tensor(primals_181, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf140)
        del primals_182
        buf141 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_146, x_149], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf140, buf141, 3211264, grid=grid(3211264), stream=stream0)
        buf142 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_184, buf141, reinterpret_tensor(primals_183, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf142)
        del primals_184
        buf143 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_10, x_153], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf142, primals_36, buf130, buf143, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_185, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf144, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf145 = buf134; del buf134  # reuse
        buf146 = buf133; del buf133  # reuse
        buf147 = buf132; del buf132  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf144, primals_186, buf145, buf146, buf147, 6272, 128, grid=grid(6272), stream=stream0)
        buf148 = buf136; del buf136  # reuse
        buf149 = buf135; del buf135  # reuse
        buf522 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf145, buf146, buf147, buf148, buf149, buf522, 1568, 4, grid=grid(1568), stream=stream0)
        buf151 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf144, primals_186, buf148, buf149, buf151, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_186
        buf152 = reinterpret_tensor(buf144, (1568, 512), (512, 1), 0); del buf144  # reuse
        # Source Nodes: [x_158, x_159], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf151, primals_37, primals_38, buf152, 802816, grid=grid(802816), stream=stream0)
        del primals_38
        buf153 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_159], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_188, buf152, reinterpret_tensor(primals_187, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf153)
        del primals_188
        buf154 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160, x_163], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf153, buf154, 3211264, grid=grid(3211264), stream=stream0)
        buf155 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_190, buf154, reinterpret_tensor(primals_189, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf155)
        del primals_190
        buf156 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_11, x_167], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf155, primals_39, buf143, buf156, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_191, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf157, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf158 = buf147; del buf147  # reuse
        buf159 = buf146; del buf146  # reuse
        buf160 = buf145; del buf145  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf157, primals_192, buf158, buf159, buf160, 6272, 128, grid=grid(6272), stream=stream0)
        buf161 = buf149; del buf149  # reuse
        buf162 = buf148; del buf148  # reuse
        buf521 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf158, buf159, buf160, buf161, buf162, buf521, 1568, 4, grid=grid(1568), stream=stream0)
        buf164 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf157, primals_192, buf161, buf162, buf164, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_192
        buf165 = reinterpret_tensor(buf157, (1568, 512), (512, 1), 0); del buf157  # reuse
        # Source Nodes: [x_172, x_173], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf164, primals_40, primals_41, buf165, 802816, grid=grid(802816), stream=stream0)
        del primals_41
        buf166 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_194, buf165, reinterpret_tensor(primals_193, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf166)
        del primals_194
        buf167 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174, x_177], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf166, buf167, 3211264, grid=grid(3211264), stream=stream0)
        buf168 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_196, buf167, reinterpret_tensor(primals_195, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf168)
        del primals_196
        buf169 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_12, x_181], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf168, primals_42, buf156, buf169, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_197, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf170, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf171 = buf160; del buf160  # reuse
        buf172 = buf159; del buf159  # reuse
        buf173 = buf158; del buf158  # reuse
        # Source Nodes: [x_186], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf170, primals_198, buf171, buf172, buf173, 6272, 128, grid=grid(6272), stream=stream0)
        buf174 = buf162; del buf162  # reuse
        buf175 = buf161; del buf161  # reuse
        buf520 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_186], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf171, buf172, buf173, buf174, buf175, buf520, 1568, 4, grid=grid(1568), stream=stream0)
        buf177 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_186], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf170, primals_198, buf174, buf175, buf177, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_198
        buf178 = reinterpret_tensor(buf170, (1568, 512), (512, 1), 0); del buf170  # reuse
        # Source Nodes: [x_186, x_187], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf177, primals_43, primals_44, buf178, 802816, grid=grid(802816), stream=stream0)
        del primals_44
        buf179 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_200, buf178, reinterpret_tensor(primals_199, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf179)
        del primals_200
        buf180 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188, x_191], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf179, buf180, 3211264, grid=grid(3211264), stream=stream0)
        buf181 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_202, buf180, reinterpret_tensor(primals_201, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf181)
        del primals_202
        buf182 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_13, x_195], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf181, primals_45, buf169, buf182, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_197], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_203, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf183, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf184 = buf173; del buf173  # reuse
        buf185 = buf172; del buf172  # reuse
        buf186 = buf171; del buf171  # reuse
        # Source Nodes: [x_200], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf183, primals_204, buf184, buf185, buf186, 6272, 128, grid=grid(6272), stream=stream0)
        buf187 = buf175; del buf175  # reuse
        buf188 = buf174; del buf174  # reuse
        buf519 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf184, buf185, buf186, buf187, buf188, buf519, 1568, 4, grid=grid(1568), stream=stream0)
        buf190 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf183, primals_204, buf187, buf188, buf190, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_204
        buf191 = reinterpret_tensor(buf183, (1568, 512), (512, 1), 0); del buf183  # reuse
        # Source Nodes: [x_200, x_201], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf190, primals_46, primals_47, buf191, 802816, grid=grid(802816), stream=stream0)
        del primals_47
        buf192 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_201], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_206, buf191, reinterpret_tensor(primals_205, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf192)
        del primals_206
        buf193 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_202, x_205], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf192, buf193, 3211264, grid=grid(3211264), stream=stream0)
        buf194 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_208, buf193, reinterpret_tensor(primals_207, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf194)
        del primals_208
        buf195 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_14, x_209], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf194, primals_48, buf182, buf195, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_211], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_209, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf196, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf197 = buf186; del buf186  # reuse
        buf198 = buf185; del buf185  # reuse
        buf199 = buf184; del buf184  # reuse
        # Source Nodes: [x_214], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf196, primals_210, buf197, buf198, buf199, 6272, 128, grid=grid(6272), stream=stream0)
        buf200 = buf188; del buf188  # reuse
        buf201 = buf187; del buf187  # reuse
        buf518 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_214], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf197, buf198, buf199, buf200, buf201, buf518, 1568, 4, grid=grid(1568), stream=stream0)
        buf203 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_214], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf196, primals_210, buf200, buf201, buf203, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_210
        buf204 = reinterpret_tensor(buf196, (1568, 512), (512, 1), 0); del buf196  # reuse
        # Source Nodes: [x_214, x_215], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf203, primals_49, primals_50, buf204, 802816, grid=grid(802816), stream=stream0)
        del primals_50
        buf205 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_212, buf204, reinterpret_tensor(primals_211, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf205)
        del primals_212
        buf206 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_216, x_219], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf205, buf206, 3211264, grid=grid(3211264), stream=stream0)
        buf207 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_219], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_214, buf206, reinterpret_tensor(primals_213, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf207)
        del primals_214
        buf208 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_15, x_223], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf207, primals_51, buf195, buf208, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_225], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_215, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf209, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf210 = buf199; del buf199  # reuse
        buf211 = buf198; del buf198  # reuse
        buf212 = buf197; del buf197  # reuse
        # Source Nodes: [x_228], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf209, primals_216, buf210, buf211, buf212, 6272, 128, grid=grid(6272), stream=stream0)
        buf213 = buf201; del buf201  # reuse
        buf214 = buf200; del buf200  # reuse
        buf517 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf210, buf211, buf212, buf213, buf214, buf517, 1568, 4, grid=grid(1568), stream=stream0)
        buf216 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf209, primals_216, buf213, buf214, buf216, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_216
        buf217 = reinterpret_tensor(buf209, (1568, 512), (512, 1), 0); del buf209  # reuse
        # Source Nodes: [x_228, x_229], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf216, primals_52, primals_53, buf217, 802816, grid=grid(802816), stream=stream0)
        del primals_53
        buf218 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_229], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_218, buf217, reinterpret_tensor(primals_217, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf218)
        del primals_218
        buf219 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_230, x_233], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf218, buf219, 3211264, grid=grid(3211264), stream=stream0)
        buf220 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_220, buf219, reinterpret_tensor(primals_219, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf220)
        del primals_220
        buf221 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_16, x_237], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf220, primals_54, buf208, buf221, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_239], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_221, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf222, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf223 = buf212; del buf212  # reuse
        buf224 = buf211; del buf211  # reuse
        buf225 = buf210; del buf210  # reuse
        # Source Nodes: [x_242], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf222, primals_222, buf223, buf224, buf225, 6272, 128, grid=grid(6272), stream=stream0)
        buf226 = buf214; del buf214  # reuse
        buf227 = buf213; del buf213  # reuse
        buf516 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf223, buf224, buf225, buf226, buf227, buf516, 1568, 4, grid=grid(1568), stream=stream0)
        buf229 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf222, primals_222, buf226, buf227, buf229, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_222
        buf230 = reinterpret_tensor(buf222, (1568, 512), (512, 1), 0); del buf222  # reuse
        # Source Nodes: [x_242, x_243], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf229, primals_55, primals_56, buf230, 802816, grid=grid(802816), stream=stream0)
        del primals_56
        buf231 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_224, buf230, reinterpret_tensor(primals_223, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf231)
        del primals_224
        buf232 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244, x_247], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf231, buf232, 3211264, grid=grid(3211264), stream=stream0)
        buf233 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_247], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_226, buf232, reinterpret_tensor(primals_225, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf233)
        del primals_226
        buf234 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_17, x_251], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf233, primals_57, buf221, buf234, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_253], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_227, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf235, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf236 = buf225; del buf225  # reuse
        buf237 = buf224; del buf224  # reuse
        buf238 = buf223; del buf223  # reuse
        # Source Nodes: [x_256], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf235, primals_228, buf236, buf237, buf238, 6272, 128, grid=grid(6272), stream=stream0)
        buf239 = buf227; del buf227  # reuse
        buf240 = buf226; del buf226  # reuse
        buf515 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf236, buf237, buf238, buf239, buf240, buf515, 1568, 4, grid=grid(1568), stream=stream0)
        buf242 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf235, primals_228, buf239, buf240, buf242, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_228
        buf243 = reinterpret_tensor(buf235, (1568, 512), (512, 1), 0); del buf235  # reuse
        # Source Nodes: [x_256, x_257], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf242, primals_58, primals_59, buf243, 802816, grid=grid(802816), stream=stream0)
        del primals_59
        buf244 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_257], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_230, buf243, reinterpret_tensor(primals_229, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf244)
        del primals_230
        buf245 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_258, x_261], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf244, buf245, 3211264, grid=grid(3211264), stream=stream0)
        buf246 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_261], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_232, buf245, reinterpret_tensor(primals_231, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf246)
        del primals_232
        buf247 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_18, x_265], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf246, primals_60, buf234, buf247, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_267], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_233, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf248, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf249 = buf238; del buf238  # reuse
        buf250 = buf237; del buf237  # reuse
        buf251 = buf236; del buf236  # reuse
        # Source Nodes: [x_270], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf248, primals_234, buf249, buf250, buf251, 6272, 128, grid=grid(6272), stream=stream0)
        buf252 = buf240; del buf240  # reuse
        buf253 = buf239; del buf239  # reuse
        buf514 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_270], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf249, buf250, buf251, buf252, buf253, buf514, 1568, 4, grid=grid(1568), stream=stream0)
        buf255 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_270], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf248, primals_234, buf252, buf253, buf255, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_234
        buf256 = reinterpret_tensor(buf248, (1568, 512), (512, 1), 0); del buf248  # reuse
        # Source Nodes: [x_270, x_271], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf255, primals_61, primals_62, buf256, 802816, grid=grid(802816), stream=stream0)
        del primals_62
        buf257 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_236, buf256, reinterpret_tensor(primals_235, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf257)
        del primals_236
        buf258 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272, x_275], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf257, buf258, 3211264, grid=grid(3211264), stream=stream0)
        buf259 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_275], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_238, buf258, reinterpret_tensor(primals_237, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf259)
        del primals_238
        buf260 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_19, x_279], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf259, primals_63, buf247, buf260, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_281], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_239, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf261, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf262 = buf251; del buf251  # reuse
        buf263 = buf250; del buf250  # reuse
        buf264 = buf249; del buf249  # reuse
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf261, primals_240, buf262, buf263, buf264, 6272, 128, grid=grid(6272), stream=stream0)
        buf265 = buf253; del buf253  # reuse
        buf266 = buf252; del buf252  # reuse
        buf513 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf262, buf263, buf264, buf265, buf266, buf513, 1568, 4, grid=grid(1568), stream=stream0)
        buf268 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf261, primals_240, buf265, buf266, buf268, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_240
        buf269 = reinterpret_tensor(buf261, (1568, 512), (512, 1), 0); del buf261  # reuse
        # Source Nodes: [x_284, x_285], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf268, primals_64, primals_65, buf269, 802816, grid=grid(802816), stream=stream0)
        del primals_65
        buf270 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_285], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_242, buf269, reinterpret_tensor(primals_241, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf270)
        del primals_242
        buf271 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_286, x_289], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf270, buf271, 3211264, grid=grid(3211264), stream=stream0)
        buf272 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_289], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_244, buf271, reinterpret_tensor(primals_243, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf272)
        del primals_244
        buf273 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_20, x_293], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf272, primals_66, buf260, buf273, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_295], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_245, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf274, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf275 = buf264; del buf264  # reuse
        buf276 = buf263; del buf263  # reuse
        buf277 = buf262; del buf262  # reuse
        # Source Nodes: [x_298], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf274, primals_246, buf275, buf276, buf277, 6272, 128, grid=grid(6272), stream=stream0)
        buf278 = buf266; del buf266  # reuse
        buf279 = buf265; del buf265  # reuse
        buf512 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf275, buf276, buf277, buf278, buf279, buf512, 1568, 4, grid=grid(1568), stream=stream0)
        buf281 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf274, primals_246, buf278, buf279, buf281, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_246
        buf282 = reinterpret_tensor(buf274, (1568, 512), (512, 1), 0); del buf274  # reuse
        # Source Nodes: [x_298, x_299], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf281, primals_67, primals_68, buf282, 802816, grid=grid(802816), stream=stream0)
        del primals_68
        buf283 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_299], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_248, buf282, reinterpret_tensor(primals_247, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf283)
        del primals_248
        buf284 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_300, x_303], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf283, buf284, 3211264, grid=grid(3211264), stream=stream0)
        buf285 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_303], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_250, buf284, reinterpret_tensor(primals_249, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf285)
        del primals_250
        buf286 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_21, x_307], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf285, primals_69, buf273, buf286, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_309], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_251, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf287, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf288 = buf277; del buf277  # reuse
        buf289 = buf276; del buf276  # reuse
        buf290 = buf275; del buf275  # reuse
        # Source Nodes: [x_312], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf287, primals_252, buf288, buf289, buf290, 6272, 128, grid=grid(6272), stream=stream0)
        buf291 = buf279; del buf279  # reuse
        buf292 = buf278; del buf278  # reuse
        buf511 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_312], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf288, buf289, buf290, buf291, buf292, buf511, 1568, 4, grid=grid(1568), stream=stream0)
        buf294 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_312], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf287, primals_252, buf291, buf292, buf294, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_252
        buf295 = reinterpret_tensor(buf287, (1568, 512), (512, 1), 0); del buf287  # reuse
        # Source Nodes: [x_312, x_313], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf294, primals_70, primals_71, buf295, 802816, grid=grid(802816), stream=stream0)
        del primals_71
        buf296 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_313], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_254, buf295, reinterpret_tensor(primals_253, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf296)
        del primals_254
        buf297 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_314, x_317], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf296, buf297, 3211264, grid=grid(3211264), stream=stream0)
        buf298 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_317], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_256, buf297, reinterpret_tensor(primals_255, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf298)
        del primals_256
        buf299 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_22, x_321], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf298, primals_72, buf286, buf299, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_323], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_257, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf300, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf301 = buf290; del buf290  # reuse
        buf302 = buf289; del buf289  # reuse
        buf303 = buf288; del buf288  # reuse
        # Source Nodes: [x_326], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf300, primals_258, buf301, buf302, buf303, 6272, 128, grid=grid(6272), stream=stream0)
        buf304 = buf292; del buf292  # reuse
        buf305 = buf291; del buf291  # reuse
        buf510 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_326], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf301, buf302, buf303, buf304, buf305, buf510, 1568, 4, grid=grid(1568), stream=stream0)
        buf307 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_326], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf300, primals_258, buf304, buf305, buf307, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_258
        buf308 = reinterpret_tensor(buf300, (1568, 512), (512, 1), 0); del buf300  # reuse
        # Source Nodes: [x_326, x_327], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf307, primals_73, primals_74, buf308, 802816, grid=grid(802816), stream=stream0)
        del primals_74
        buf309 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_327], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_260, buf308, reinterpret_tensor(primals_259, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf309)
        del primals_260
        buf310 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_328, x_331], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf309, buf310, 3211264, grid=grid(3211264), stream=stream0)
        buf311 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_331], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_262, buf310, reinterpret_tensor(primals_261, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf311)
        del primals_262
        buf312 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_23, x_335], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf311, primals_75, buf299, buf312, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_337], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, primals_263, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf313, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf314 = buf303; del buf303  # reuse
        buf315 = buf302; del buf302  # reuse
        buf316 = buf301; del buf301  # reuse
        # Source Nodes: [x_340], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf313, primals_264, buf314, buf315, buf316, 6272, 128, grid=grid(6272), stream=stream0)
        buf317 = buf305; del buf305  # reuse
        buf318 = buf304; del buf304  # reuse
        buf509 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_340], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf314, buf315, buf316, buf317, buf318, buf509, 1568, 4, grid=grid(1568), stream=stream0)
        buf320 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_340], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf313, primals_264, buf317, buf318, buf320, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_264
        buf321 = reinterpret_tensor(buf313, (1568, 512), (512, 1), 0); del buf313  # reuse
        # Source Nodes: [x_340, x_341], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf320, primals_76, primals_77, buf321, 802816, grid=grid(802816), stream=stream0)
        del primals_77
        buf322 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_341], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_266, buf321, reinterpret_tensor(primals_265, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf322)
        del primals_266
        buf323 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_342, x_345], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf322, buf323, 3211264, grid=grid(3211264), stream=stream0)
        buf324 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_345], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_268, buf323, reinterpret_tensor(primals_267, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf324)
        del primals_268
        buf325 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_24, x_349], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf324, primals_78, buf312, buf325, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_351], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_269, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf326, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf327 = buf316; del buf316  # reuse
        buf328 = buf315; del buf315  # reuse
        buf329 = buf314; del buf314  # reuse
        # Source Nodes: [x_354], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf326, primals_270, buf327, buf328, buf329, 6272, 128, grid=grid(6272), stream=stream0)
        buf330 = buf318; del buf318  # reuse
        buf331 = buf317; del buf317  # reuse
        buf508 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf327, buf328, buf329, buf330, buf331, buf508, 1568, 4, grid=grid(1568), stream=stream0)
        buf333 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf326, primals_270, buf330, buf331, buf333, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_270
        buf334 = reinterpret_tensor(buf326, (1568, 512), (512, 1), 0); del buf326  # reuse
        # Source Nodes: [x_354, x_355], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf333, primals_79, primals_80, buf334, 802816, grid=grid(802816), stream=stream0)
        del primals_80
        buf335 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_355], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_272, buf334, reinterpret_tensor(primals_271, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf335)
        del primals_272
        buf336 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_356, x_359], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf335, buf336, 3211264, grid=grid(3211264), stream=stream0)
        buf337 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_359], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_274, buf336, reinterpret_tensor(primals_273, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf337)
        del primals_274
        buf338 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_25, x_363], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf337, primals_81, buf325, buf338, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_365], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_275, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf339, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf340 = buf329; del buf329  # reuse
        buf341 = buf328; del buf328  # reuse
        buf342 = buf327; del buf327  # reuse
        # Source Nodes: [x_368], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf339, primals_276, buf340, buf341, buf342, 6272, 128, grid=grid(6272), stream=stream0)
        buf343 = buf331; del buf331  # reuse
        buf344 = buf330; del buf330  # reuse
        buf507 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_368], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf340, buf341, buf342, buf343, buf344, buf507, 1568, 4, grid=grid(1568), stream=stream0)
        buf346 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_368], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf339, primals_276, buf343, buf344, buf346, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_276
        buf347 = reinterpret_tensor(buf339, (1568, 512), (512, 1), 0); del buf339  # reuse
        # Source Nodes: [x_368, x_369], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf346, primals_82, primals_83, buf347, 802816, grid=grid(802816), stream=stream0)
        del primals_83
        buf348 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_369], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_278, buf347, reinterpret_tensor(primals_277, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf348)
        del primals_278
        buf349 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_370, x_373], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf348, buf349, 3211264, grid=grid(3211264), stream=stream0)
        buf350 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_373], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_280, buf349, reinterpret_tensor(primals_279, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf350)
        del primals_280
        buf351 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_26, x_377], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf350, primals_84, buf338, buf351, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_379], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_281, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf352, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf353 = buf342; del buf342  # reuse
        buf354 = buf341; del buf341  # reuse
        buf355 = buf340; del buf340  # reuse
        # Source Nodes: [x_382], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf352, primals_282, buf353, buf354, buf355, 6272, 128, grid=grid(6272), stream=stream0)
        buf356 = buf344; del buf344  # reuse
        buf357 = buf343; del buf343  # reuse
        buf506 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_382], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf353, buf354, buf355, buf356, buf357, buf506, 1568, 4, grid=grid(1568), stream=stream0)
        buf359 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_382], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf352, primals_282, buf356, buf357, buf359, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_282
        buf360 = reinterpret_tensor(buf352, (1568, 512), (512, 1), 0); del buf352  # reuse
        # Source Nodes: [x_382, x_383], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf359, primals_85, primals_86, buf360, 802816, grid=grid(802816), stream=stream0)
        del primals_86
        buf361 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_383], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_284, buf360, reinterpret_tensor(primals_283, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf361)
        del primals_284
        buf362 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_384, x_387], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf361, buf362, 3211264, grid=grid(3211264), stream=stream0)
        buf363 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_387], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_286, buf362, reinterpret_tensor(primals_285, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf363)
        del primals_286
        buf364 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_27, x_391], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf363, primals_87, buf351, buf364, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_393], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, primals_287, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf365, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf366 = buf355; del buf355  # reuse
        buf367 = buf354; del buf354  # reuse
        buf368 = buf353; del buf353  # reuse
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf365, primals_288, buf366, buf367, buf368, 6272, 128, grid=grid(6272), stream=stream0)
        buf369 = buf357; del buf357  # reuse
        buf370 = buf356; del buf356  # reuse
        buf505 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf366, buf367, buf368, buf369, buf370, buf505, 1568, 4, grid=grid(1568), stream=stream0)
        buf372 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf365, primals_288, buf369, buf370, buf372, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_288
        buf373 = reinterpret_tensor(buf365, (1568, 512), (512, 1), 0); del buf365  # reuse
        # Source Nodes: [x_396, x_397], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf372, primals_88, primals_89, buf373, 802816, grid=grid(802816), stream=stream0)
        del primals_89
        buf374 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_397], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_290, buf373, reinterpret_tensor(primals_289, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf374)
        del primals_290
        buf375 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_398, x_401], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf374, buf375, 3211264, grid=grid(3211264), stream=stream0)
        buf376 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_401], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_292, buf375, reinterpret_tensor(primals_291, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf376)
        del primals_292
        buf377 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_28, x_405], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf376, primals_90, buf364, buf377, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_407], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_293, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf378, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf379 = buf368; del buf368  # reuse
        buf380 = buf367; del buf367  # reuse
        buf381 = buf366; del buf366  # reuse
        # Source Nodes: [x_410], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf378, primals_294, buf379, buf380, buf381, 6272, 128, grid=grid(6272), stream=stream0)
        buf382 = buf370; del buf370  # reuse
        buf383 = buf369; del buf369  # reuse
        buf504 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_410], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf379, buf380, buf381, buf382, buf383, buf504, 1568, 4, grid=grid(1568), stream=stream0)
        buf385 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_410], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf378, primals_294, buf382, buf383, buf385, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_294
        buf386 = reinterpret_tensor(buf378, (1568, 512), (512, 1), 0); del buf378  # reuse
        # Source Nodes: [x_410, x_411], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf385, primals_91, primals_92, buf386, 802816, grid=grid(802816), stream=stream0)
        del primals_92
        buf387 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_411], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_296, buf386, reinterpret_tensor(primals_295, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf387)
        del primals_296
        buf388 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_412, x_415], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf387, buf388, 3211264, grid=grid(3211264), stream=stream0)
        buf389 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_415], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_298, buf388, reinterpret_tensor(primals_297, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf389)
        del primals_298
        buf390 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_29, x_419], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf389, primals_93, buf377, buf390, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_421], Original ATen: [aten.convolution]
        buf391 = extern_kernels.convolution(buf390, primals_299, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf391, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf392 = buf381; del buf381  # reuse
        buf393 = buf380; del buf380  # reuse
        buf394 = buf379; del buf379  # reuse
        # Source Nodes: [x_424], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf391, primals_300, buf392, buf393, buf394, 6272, 128, grid=grid(6272), stream=stream0)
        buf395 = buf383; del buf383  # reuse
        buf396 = buf382; del buf382  # reuse
        buf503 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_424], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf392, buf393, buf394, buf395, buf396, buf503, 1568, 4, grid=grid(1568), stream=stream0)
        buf398 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_424], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf391, primals_300, buf395, buf396, buf398, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_300
        buf399 = reinterpret_tensor(buf391, (1568, 512), (512, 1), 0); del buf391  # reuse
        # Source Nodes: [x_424, x_425], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf398, primals_94, primals_95, buf399, 802816, grid=grid(802816), stream=stream0)
        del primals_95
        buf400 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_425], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_302, buf399, reinterpret_tensor(primals_301, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf400)
        del primals_302
        buf401 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_426, x_429], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf400, buf401, 3211264, grid=grid(3211264), stream=stream0)
        buf402 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_429], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_304, buf401, reinterpret_tensor(primals_303, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf402)
        del primals_304
        buf403 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_30, x_433], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf402, primals_96, buf390, buf403, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_435], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_305, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf404, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf405 = buf394; del buf394  # reuse
        buf406 = buf393; del buf393  # reuse
        buf407 = buf392; del buf392  # reuse
        # Source Nodes: [x_438], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf404, primals_306, buf405, buf406, buf407, 6272, 128, grid=grid(6272), stream=stream0)
        buf408 = buf396; del buf396  # reuse
        buf409 = buf395; del buf395  # reuse
        buf502 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_438], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf405, buf406, buf407, buf408, buf409, buf502, 1568, 4, grid=grid(1568), stream=stream0)
        buf411 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_438], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf404, primals_306, buf408, buf409, buf411, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_306
        buf412 = reinterpret_tensor(buf404, (1568, 512), (512, 1), 0); del buf404  # reuse
        # Source Nodes: [x_438, x_439], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf411, primals_97, primals_98, buf412, 802816, grid=grid(802816), stream=stream0)
        del primals_98
        buf413 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_439], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_308, buf412, reinterpret_tensor(primals_307, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf413)
        del primals_308
        buf414 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_440, x_443], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf413, buf414, 3211264, grid=grid(3211264), stream=stream0)
        buf415 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_443], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_310, buf414, reinterpret_tensor(primals_309, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf415)
        del primals_310
        buf416 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_31, x_447], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf415, primals_99, buf403, buf416, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_449], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf416, primals_311, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf417, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf418 = buf407; del buf407  # reuse
        buf419 = buf406; del buf406  # reuse
        buf420 = buf405; del buf405  # reuse
        # Source Nodes: [x_452], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf417, primals_312, buf418, buf419, buf420, 6272, 128, grid=grid(6272), stream=stream0)
        buf421 = buf409; del buf409  # reuse
        buf422 = buf408; del buf408  # reuse
        buf501 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_452], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf418, buf419, buf420, buf421, buf422, buf501, 1568, 4, grid=grid(1568), stream=stream0)
        buf424 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_452], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf417, primals_312, buf421, buf422, buf424, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_312
        buf425 = reinterpret_tensor(buf417, (1568, 512), (512, 1), 0); del buf417  # reuse
        # Source Nodes: [x_452, x_453], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf424, primals_100, primals_101, buf425, 802816, grid=grid(802816), stream=stream0)
        del primals_101
        buf426 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_453], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_314, buf425, reinterpret_tensor(primals_313, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf426)
        del primals_314
        buf427 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_454, x_457], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf426, buf427, 3211264, grid=grid(3211264), stream=stream0)
        buf428 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_457], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_316, buf427, reinterpret_tensor(primals_315, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf428)
        del primals_316
        buf429 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_32, x_461], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf428, primals_102, buf416, buf429, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_463], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, primals_317, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf430, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf431 = buf420; del buf420  # reuse
        buf432 = buf419; del buf419  # reuse
        buf433 = buf418; del buf418  # reuse
        # Source Nodes: [x_466], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf430, primals_318, buf431, buf432, buf433, 6272, 128, grid=grid(6272), stream=stream0)
        buf434 = buf422; del buf422  # reuse
        buf435 = buf421; del buf421  # reuse
        buf500 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_466], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_25.run(buf431, buf432, buf433, buf434, buf435, buf500, 1568, 4, grid=grid(1568), stream=stream0)
        del buf431
        del buf432
        del buf433
        buf437 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_466], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf430, primals_318, buf434, buf435, buf437, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del primals_318
        buf438 = reinterpret_tensor(buf430, (1568, 512), (512, 1), 0); del buf430  # reuse
        # Source Nodes: [x_466, x_467], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf437, primals_103, primals_104, buf438, 802816, grid=grid(802816), stream=stream0)
        del primals_104
        buf439 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_467], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_320, buf438, reinterpret_tensor(primals_319, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf439)
        del primals_320
        buf440 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_468, x_471], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_28.run(buf439, buf440, 3211264, grid=grid(3211264), stream=stream0)
        buf441 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_471], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_322, buf440, reinterpret_tensor(primals_321, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf441)
        del primals_322
        buf442 = buf435; del buf435  # reuse
        buf443 = buf434; del buf434  # reuse
        buf499 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_479], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_30.run(buf441, primals_105, buf429, buf442, buf443, buf499, 1568, 512, grid=grid(1568), stream=stream0)
        buf445 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_479], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_31.run(buf441, primals_105, buf429, buf442, buf443, buf445, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del buf442
        del buf443
        buf446 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_479, x_480], Original ATen: [aten.native_layer_norm, aten.permute]
        triton_poi_fused_native_layer_norm_permute_32.run(buf445, primals_106, primals_107, buf446, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del primals_107
        # Source Nodes: [shortcut_33], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, buf3, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf448 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf447, primals_324, buf448, 8192, 49, grid=grid(8192, 49), stream=stream0)
        del primals_324
        # Source Nodes: [x_482], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf448, primals_325, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf449, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf450 = empty_strided((8, 7, 7, 1, 8), (392, 7, 1, 3136, 49), device='cuda', dtype=torch.float32)
        buf451 = empty_strided((8, 7, 7, 1, 8), (392, 7, 1, 3136, 49), device='cuda', dtype=torch.float32)
        buf452 = empty_strided((8, 7, 7, 1, 8), (392, 7, 1, 3136, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_485], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_34.run(buf449, primals_326, buf450, buf451, buf452, 3136, 128, grid=grid(3136), stream=stream0)
        buf453 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cuda', dtype=torch.float32)
        buf454 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cuda', dtype=torch.float32)
        buf498 = empty_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_485], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_35.run(buf450, buf451, buf452, buf453, buf454, buf498, 392, 8, grid=grid(392), stream=stream0)
        buf456 = reinterpret_tensor(buf447, (8, 7, 7, 1024), (50176, 1, 7168, 7), 0); del buf447  # reuse
        # Source Nodes: [x_485], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_36.run(buf449, primals_326, buf453, buf454, buf456, 56, 7168, grid=grid(56, 7168), stream=stream0)
        del primals_326
        buf457 = reinterpret_tensor(buf449, (392, 1024), (1024, 1), 0); del buf449  # reuse
        # Source Nodes: [x_485, x_486], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_37.run(buf456, primals_108, primals_109, buf457, 401408, grid=grid(401408), stream=stream0)
        del primals_109
        buf458 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_486], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_328, buf457, reinterpret_tensor(primals_327, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf458)
        del primals_328
        buf459 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_487, x_490], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf458, buf459, 1605632, grid=grid(1605632), stream=stream0)
        buf460 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_490], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_330, buf459, reinterpret_tensor(primals_329, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf460)
        del primals_330
        buf461 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_34, x_494], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_39.run(buf460, primals_110, buf448, buf461, 401408, grid=grid(401408), stream=stream0)
        # Source Nodes: [x_496], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_331, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf462, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf463 = buf452; del buf452  # reuse
        buf464 = buf451; del buf451  # reuse
        buf465 = buf450; del buf450  # reuse
        # Source Nodes: [x_499], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_34.run(buf462, primals_332, buf463, buf464, buf465, 3136, 128, grid=grid(3136), stream=stream0)
        buf466 = buf454; del buf454  # reuse
        buf467 = buf453; del buf453  # reuse
        buf497 = empty_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_499], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_35.run(buf463, buf464, buf465, buf466, buf467, buf497, 392, 8, grid=grid(392), stream=stream0)
        buf469 = empty_strided((8, 7, 7, 1024), (50176, 1, 7168, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_499], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_36.run(buf462, primals_332, buf466, buf467, buf469, 56, 7168, grid=grid(56, 7168), stream=stream0)
        del primals_332
        buf470 = reinterpret_tensor(buf462, (392, 1024), (1024, 1), 0); del buf462  # reuse
        # Source Nodes: [x_499, x_500], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_37.run(buf469, primals_111, primals_112, buf470, 401408, grid=grid(401408), stream=stream0)
        del primals_112
        buf471 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_500], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_334, buf470, reinterpret_tensor(primals_333, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf471)
        del primals_334
        buf472 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_501, x_504], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf471, buf472, 1605632, grid=grid(1605632), stream=stream0)
        buf473 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_504], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_336, buf472, reinterpret_tensor(primals_335, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf473)
        del primals_336
        buf474 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_35, x_508], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_39.run(buf473, primals_113, buf461, buf474, 401408, grid=grid(401408), stream=stream0)
        # Source Nodes: [x_510], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf474, primals_337, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf475, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf476 = buf465; del buf465  # reuse
        buf477 = buf464; del buf464  # reuse
        buf478 = buf463; del buf463  # reuse
        # Source Nodes: [x_513], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_34.run(buf475, primals_338, buf476, buf477, buf478, 3136, 128, grid=grid(3136), stream=stream0)
        buf479 = buf467; del buf467  # reuse
        buf480 = buf466; del buf466  # reuse
        buf496 = empty_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_513], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_35.run(buf476, buf477, buf478, buf479, buf480, buf496, 392, 8, grid=grid(392), stream=stream0)
        del buf476
        del buf477
        del buf478
        buf482 = empty_strided((8, 7, 7, 1024), (50176, 1, 7168, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_513], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_36.run(buf475, primals_338, buf479, buf480, buf482, 56, 7168, grid=grid(56, 7168), stream=stream0)
        del buf479
        del buf480
        del primals_338
        buf483 = reinterpret_tensor(buf475, (392, 1024), (1024, 1), 0); del buf475  # reuse
        # Source Nodes: [x_513, x_514], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_37.run(buf482, primals_114, primals_115, buf483, 401408, grid=grid(401408), stream=stream0)
        del primals_115
        buf484 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_514], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_340, buf483, reinterpret_tensor(primals_339, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf484)
        del primals_340
        buf485 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_515, x_518], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf484, buf485, 1605632, grid=grid(1605632), stream=stream0)
        buf486 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_518], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_342, buf485, reinterpret_tensor(primals_341, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf486)
        del primals_342
        buf487 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf488 = reinterpret_tensor(buf487, (8, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf487  # reuse
        # Source Nodes: [x_522, x_525, x_528], Original ATen: [aten.add, aten.mean, aten.mul]
        triton_per_fused_add_mean_mul_40.run(buf488, buf486, primals_116, buf474, 8192, 49, grid=grid(8192), stream=stream0)
        buf492 = empty_strided((8, 1, 1, 1024), (1024, 1, 1024, 1), device='cuda', dtype=torch.float32)
        buf493 = empty((8, 1024), device='cuda', dtype=torch.float32)
        buf495 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_532, x_535], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_41.run(buf488, primals_117, primals_118, buf492, buf493, buf495, 8, 1024, grid=grid(8), stream=stream0)
        del buf488
        del primals_118
        buf494 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_344, buf493, reinterpret_tensor(primals_343, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf494)
        del primals_344
        return (buf494, primals_1, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_110, primals_111, primals_113, primals_114, primals_116, primals_117, buf0, primals_121, primals_127, primals_133, buf1, primals_141, primals_147, primals_153, buf2, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_305, primals_311, primals_317, buf3, primals_325, primals_331, primals_337, buf4, buf9, buf10, buf15, buf16, buf17, buf18, buf19, buf20, buf25, buf26, buf27, buf28, buf29, buf30, buf35, buf36, buf37, buf38, buf39, buf43, buf44, buf46, buf54, buf55, buf56, buf57, buf58, buf59, buf67, buf68, buf69, buf70, buf71, buf72, buf80, buf81, buf82, buf83, buf84, buf88, buf89, buf91, buf99, buf100, buf101, buf102, buf103, buf104, buf112, buf113, buf114, buf115, buf116, buf117, buf125, buf126, buf127, buf128, buf129, buf130, buf138, buf139, buf140, buf141, buf142, buf143, buf151, buf152, buf153, buf154, buf155, buf156, buf164, buf165, buf166, buf167, buf168, buf169, buf177, buf178, buf179, buf180, buf181, buf182, buf190, buf191, buf192, buf193, buf194, buf195, buf203, buf204, buf205, buf206, buf207, buf208, buf216, buf217, buf218, buf219, buf220, buf221, buf229, buf230, buf231, buf232, buf233, buf234, buf242, buf243, buf244, buf245, buf246, buf247, buf255, buf256, buf257, buf258, buf259, buf260, buf268, buf269, buf270, buf271, buf272, buf273, buf281, buf282, buf283, buf284, buf285, buf286, buf294, buf295, buf296, buf297, buf298, buf299, buf307, buf308, buf309, buf310, buf311, buf312, buf320, buf321, buf322, buf323, buf324, buf325, buf333, buf334, buf335, buf336, buf337, buf338, buf346, buf347, buf348, buf349, buf350, buf351, buf359, buf360, buf361, buf362, buf363, buf364, buf372, buf373, buf374, buf375, buf376, buf377, buf385, buf386, buf387, buf388, buf389, buf390, buf398, buf399, buf400, buf401, buf402, buf403, buf411, buf412, buf413, buf414, buf415, buf416, buf424, buf425, buf426, buf427, buf428, buf429, buf437, buf438, buf439, buf440, buf441, buf445, buf446, buf448, buf456, buf457, buf458, buf459, buf460, buf461, buf469, buf470, buf471, buf472, buf473, buf474, buf482, buf483, buf484, buf485, buf486, buf492, buf493, reinterpret_tensor(primals_343, (1000, 1024), (1024, 1), 0), buf495, reinterpret_tensor(primals_341, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_339, (4096, 1024), (1024, 1), 0), buf496, reinterpret_tensor(primals_335, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_333, (4096, 1024), (1024, 1), 0), buf497, reinterpret_tensor(primals_329, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_327, (4096, 1024), (1024, 1), 0), buf498, buf499, reinterpret_tensor(primals_321, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_319, (2048, 512), (512, 1), 0), buf500, reinterpret_tensor(primals_315, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_313, (2048, 512), (512, 1), 0), buf501, reinterpret_tensor(primals_309, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_307, (2048, 512), (512, 1), 0), buf502, reinterpret_tensor(primals_303, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_301, (2048, 512), (512, 1), 0), buf503, reinterpret_tensor(primals_297, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_295, (2048, 512), (512, 1), 0), buf504, reinterpret_tensor(primals_291, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_289, (2048, 512), (512, 1), 0), buf505, reinterpret_tensor(primals_285, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_283, (2048, 512), (512, 1), 0), buf506, reinterpret_tensor(primals_279, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_277, (2048, 512), (512, 1), 0), buf507, reinterpret_tensor(primals_273, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_271, (2048, 512), (512, 1), 0), buf508, reinterpret_tensor(primals_267, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_265, (2048, 512), (512, 1), 0), buf509, reinterpret_tensor(primals_261, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_259, (2048, 512), (512, 1), 0), buf510, reinterpret_tensor(primals_255, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_253, (2048, 512), (512, 1), 0), buf511, reinterpret_tensor(primals_249, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_247, (2048, 512), (512, 1), 0), buf512, reinterpret_tensor(primals_243, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_241, (2048, 512), (512, 1), 0), buf513, reinterpret_tensor(primals_237, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_235, (2048, 512), (512, 1), 0), buf514, reinterpret_tensor(primals_231, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_229, (2048, 512), (512, 1), 0), buf515, reinterpret_tensor(primals_225, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_223, (2048, 512), (512, 1), 0), buf516, reinterpret_tensor(primals_219, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_217, (2048, 512), (512, 1), 0), buf517, reinterpret_tensor(primals_213, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_211, (2048, 512), (512, 1), 0), buf518, reinterpret_tensor(primals_207, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_205, (2048, 512), (512, 1), 0), buf519, reinterpret_tensor(primals_201, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_199, (2048, 512), (512, 1), 0), buf520, reinterpret_tensor(primals_195, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_193, (2048, 512), (512, 1), 0), buf521, reinterpret_tensor(primals_189, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_187, (2048, 512), (512, 1), 0), buf522, reinterpret_tensor(primals_183, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_181, (2048, 512), (512, 1), 0), buf523, reinterpret_tensor(primals_177, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_175, (2048, 512), (512, 1), 0), buf524, reinterpret_tensor(primals_171, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_169, (2048, 512), (512, 1), 0), buf525, reinterpret_tensor(primals_165, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_163, (2048, 512), (512, 1), 0), buf526, buf527, reinterpret_tensor(primals_157, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_155, (1024, 256), (256, 1), 0), buf528, reinterpret_tensor(primals_151, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_149, (1024, 256), (256, 1), 0), buf529, reinterpret_tensor(primals_145, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_143, (1024, 256), (256, 1), 0), buf530, buf531, reinterpret_tensor(primals_137, (128, 512), (512, 1), 0), reinterpret_tensor(primals_135, (512, 128), (128, 1), 0), buf532, reinterpret_tensor(primals_131, (128, 512), (512, 1), 0), reinterpret_tensor(primals_129, (512, 128), (128, 1), 0), buf533, reinterpret_tensor(primals_125, (128, 512), (512, 1), 0), reinterpret_tensor(primals_123, (512, 128), (128, 1), 0), buf534, buf535, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((512, 256, 2, 2), (1024, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((1024, 512, 2, 2), (2048, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convnext_base', benchmark_compiled_module)
