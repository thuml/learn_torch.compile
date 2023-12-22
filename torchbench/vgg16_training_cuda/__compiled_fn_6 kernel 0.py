
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


# kernel path: /tmp/torchinductor_youkaichao/gg/cgg463u26szk2koomsmwdhziq76kysbb7oomrurjsyrliqijjwz4.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1000 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (2000 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (3000 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/mw/cmww2p7xzn5getuabkwmein5fonbzzvbyl2yxmwnnm2kvz77fydz.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.int1)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(in_out_ptr0 + (x0), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbe2bkrvlj6avaws7ousyxdzcv2u2jqrqkoldsmyf3a4dblzzi6m.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (4096 + x0), None)
    tmp3 = tl.load(in_ptr0 + (8192 + x0), None)
    tmp5 = tl.load(in_ptr0 + (12288 + x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/is/cisn75gzckkqnnjmkvagakbwqkfhkdiyhxioqehxtghgeeet6snr.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceag43irddsjxc3hq37jyqmxmhhonzyc75uajv77kogbhwc7irov.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3584
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (57344*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvnbqmjzosu2cpofkribymhry2gssrsfmc6edjdf6ix5vx7yhik.py
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
    size_hints=[512, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 7
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


# kernel path: /tmp/torchinductor_youkaichao/id/cidqubou2h2bjht5zdzgfjo7pehcwbp5mbgzfbm5qu45ual5h7ob.py
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
    size_hints=[1024, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c733zowhqowskk522pv53ak77qdmcdcfmkxyotdtxo5shsmf5n6n.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camw42ez24xvnbi5khau3oxej2qhjiqda2sdseao7g2btcqvnpc5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccryuus3gdpwvpung4ftsmkazu7jjf7wfuakjl7bovt7esfd3ezy.py
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
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_youkaichao/ni/cnic4k5nlmg6ay6kuv4cutvny65fxcuurbdbr3hney3wqkgjwaeo.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p3/cp3rjrb7rjbaggin2mxip25plgw3peud54kxksde7tdcnsi37tma.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73f4l2zwoh44nzzqyu742mgnttzwht4malweghv4rxmb2rkroq4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxedu72etgmzkaighgidm3i4jkq3kgdi5iaalqabsez44lhievg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ln/clnjwt5rdsag5rp64ufmm65tsov5twrrfbizbwovyrewah7xkmwa.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3st3dg2daymiujnruc2a3osuc6cubftmv2bcusugxypsddcatv.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jq/cjqr76ttpnndx7fnqg4iyi2gw2ic3gffbsdw7e6s53a5swzkcvct.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ss/cssqqjviblntmw7s3st4fy46j42jst4s44t6vtaka74gvtmcoxpt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 392
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7gx7udltkbwmxfqcumddwvk3ec2eiubchubbksokah2p67ad2g.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (1605632*y1)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2arxm43ntukvs35w75tneind5qu6iyguyagm3sdooqbobdl2tu.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cps64a4g4z7obzwxowrlzdzvze6idbu7oogxmxt4p6gu4n2nvhmp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 196
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (12544*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4k3lbofzi3ol4b2wyuklvqplcxryz5h2gu3beqzuuga7aazs7o.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4hphhhkqjdwmdhv3mjgrxfcftfxt4zkmheauiv7n3p3prhbbzy.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 200704
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 50176
    y1 = (yindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (50176*x2) + (3211264*y1)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp4, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_33, relu, relu_1, getitem, getitem_1, relu_2, relu_3, getitem_2, getitem_3, relu_4, relu_5, relu_6, getitem_4, getitem_5, relu_7, relu_8, relu_9, getitem_6, getitem_7, relu_10, relu_11, relu_12, getitem_8, getitem_9, view, clone, clone_1, permute_3, le, permute_7, le_1, permute_11, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_3, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_5, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_7, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_9, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_11, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_13, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_15, (512, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_17, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_19, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_21, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_23, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_25, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_33, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(relu, (4, 64, 224, 224), (3211264, 1, 14336, 64))
    assert_size_stride(relu_1, (4, 64, 224, 224), (3211264, 1, 14336, 64))
    assert_size_stride(getitem, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(getitem_1, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(relu_2, (4, 128, 112, 112), (1605632, 1, 14336, 128))
    assert_size_stride(relu_3, (4, 128, 112, 112), (1605632, 1, 14336, 128))
    assert_size_stride(getitem_2, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(getitem_3, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_4, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_5, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_6, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(getitem_4, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(getitem_5, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(relu_7, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_8, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_9, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(getitem_6, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_7, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_10, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_11, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_12, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_8, (4, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(getitem_9, (4, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(view, (4, 25088), (25088, 1))
    assert_size_stride(clone, (4, 4096), (4096, 1))
    assert_size_stride(clone_1, (4, 4096), (4096, 1))
    assert_size_stride(permute_3, (1000, 4096), (4096, 1))
    assert_size_stride(le, (4, 4096), (4096, 1))
    assert_size_stride(permute_7, (4096, 4096), (4096, 1))
    assert_size_stride(le_1, (4, 4096), (4096, 1))
    assert_size_stride(permute_11, (4096, 25088), (25088, 1))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_3, out=buf0)
        del permute_3
        buf1 = empty((1000, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), clone_1, out=buf1)
        del clone_1
        buf2 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_view_0.run(tangents_1, buf2, 1000, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = buf0; del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf3, le, 16384, grid=grid(16384), stream=stream0)
        del le
        buf4 = empty((4, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, permute_7, out=buf4)
        del permute_7
        buf5 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 4), (1, 4096), 0), clone, out=buf5)
        del clone
        buf6 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_2.run(buf3, buf6, 4096, grid=grid(4096), stream=stream0)
        del buf3
        buf7 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf7, le_1, 16384, grid=grid(16384), stream=stream0)
        del le_1
        buf8 = empty((4, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf7, permute_11, out=buf8)
        del permute_11
        buf9 = empty((4096, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (4096, 4), (1, 4096), 0), view, out=buf9)
        del view
        buf10 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_2.run(buf7, buf10, 4096, grid=grid(4096), stream=stream0)
        del buf7
        # Source Nodes: [], Original ATen: [aten._adaptive_avg_pool2d_backward]
        buf11 = aten._adaptive_avg_pool2d_backward(reinterpret_tensor(buf8, (4, 512, 7, 7), (25088, 49, 7, 1), 0), getitem_8)
        del buf8
        del getitem_8
        buf12 = buf11
        del buf11
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf13 = aten.max_pool2d_with_indices_backward(buf12, relu_12, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_9)
        del buf12
        del getitem_9
        buf14 = buf13
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_3.run(buf15, relu_12, 401408, grid=grid(401408), stream=stream0)
        del relu_12
        buf16 = empty_strided((512, 7), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_4.run(buf15, buf16, 3584, 112, grid=grid(3584), stream=stream0)
        buf17 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf16, buf17, 512, 7, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf18 = aten.convolution_backward(buf15, relu_11, primals_25, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_25
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf21 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(relu_11, buf19, buf21, 784, 512, grid=grid(784, 512), stream=stream0)
        del buf19
        del relu_11
        buf22 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_4.run(buf21, buf22, 3584, 112, grid=grid(3584), stream=stream0)
        buf23 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf22, buf23, 512, 7, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf24 = aten.convolution_backward(buf21, relu_10, primals_23, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_23
        buf25 = buf24[0]
        buf26 = buf24[1]
        del buf24
        buf27 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(relu_10, buf25, buf27, 784, 512, grid=grid(784, 512), stream=stream0)
        del buf25
        del relu_10
        buf28 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_4.run(buf27, buf28, 3584, 112, grid=grid(3584), stream=stream0)
        buf29 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf28, buf29, 512, 7, grid=grid(512), stream=stream0)
        del buf28
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf30 = aten.convolution_backward(buf27, getitem_6, primals_21, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf27
        del getitem_6
        del primals_21
        buf31 = buf30[0]
        buf32 = buf30[1]
        del buf30
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf33 = aten.max_pool2d_with_indices_backward(buf31, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_7)
        del buf31
        del getitem_7
        buf34 = buf33
        del buf33
        buf35 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_7.run(buf35, relu_9, 1605632, grid=grid(1605632), stream=stream0)
        del relu_9
        buf36 = empty_strided((512, 25), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_8.run(buf35, buf36, 12800, 126, grid=grid(12800), stream=stream0)
        buf37 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_9.run(buf36, buf37, 512, 25, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf38 = aten.convolution_backward(buf35, relu_8, primals_19, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_19
        buf39 = buf38[0]
        buf40 = buf38[1]
        del buf38
        buf41 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_10.run(relu_8, buf39, buf41, 3136, 512, grid=grid(3136, 512), stream=stream0)
        del buf39
        del relu_8
        buf42 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_8.run(buf41, buf42, 12800, 126, grid=grid(12800), stream=stream0)
        buf43 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_9.run(buf42, buf43, 512, 25, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf44 = aten.convolution_backward(buf41, relu_7, primals_17, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_17
        buf45 = buf44[0]
        buf46 = buf44[1]
        del buf44
        buf47 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_10.run(relu_7, buf45, buf47, 3136, 512, grid=grid(3136, 512), stream=stream0)
        del buf45
        del relu_7
        buf48 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_8.run(buf47, buf48, 12800, 126, grid=grid(12800), stream=stream0)
        buf49 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_9.run(buf48, buf49, 512, 25, grid=grid(512), stream=stream0)
        del buf48
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf50 = aten.convolution_backward(buf47, getitem_4, primals_15, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf47
        del getitem_4
        del primals_15
        buf51 = buf50[0]
        buf52 = buf50[1]
        del buf50
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf53 = aten.max_pool2d_with_indices_backward(buf51, relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_5)
        del buf51
        del getitem_5
        buf54 = buf53
        del buf53
        buf55 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_11.run(buf55, relu_6, 3211264, grid=grid(3211264), stream=stream0)
        del relu_6
        buf56 = empty_strided((256, 98), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_12.run(buf55, buf56, 25088, 128, grid=grid(25088), stream=stream0)
        buf57 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_13.run(buf56, buf57, 256, 98, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf58 = aten.convolution_backward(buf55, relu_5, primals_13, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_13
        buf59 = buf58[0]
        buf60 = buf58[1]
        del buf58
        buf61 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_14.run(relu_5, buf59, buf61, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf59
        del relu_5
        buf62 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_12.run(buf61, buf62, 25088, 128, grid=grid(25088), stream=stream0)
        buf63 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_13.run(buf62, buf63, 256, 98, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf64 = aten.convolution_backward(buf61, relu_4, primals_11, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_11
        buf65 = buf64[0]
        buf66 = buf64[1]
        del buf64
        buf67 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_14.run(relu_4, buf65, buf67, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf65
        del relu_4
        buf68 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_12.run(buf67, buf68, 25088, 128, grid=grid(25088), stream=stream0)
        buf69 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_13.run(buf68, buf69, 256, 98, grid=grid(256), stream=stream0)
        del buf68
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf70 = aten.convolution_backward(buf67, getitem_2, primals_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf67
        del getitem_2
        del primals_9
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf73 = aten.max_pool2d_with_indices_backward(buf71, relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_3)
        del buf71
        del getitem_3
        buf74 = buf73
        del buf73
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_15.run(buf75, relu_3, 6422528, grid=grid(6422528), stream=stream0)
        del relu_3
        buf76 = empty_strided((128, 392), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_16.run(buf75, buf76, 50176, 128, grid=grid(50176), stream=stream0)
        buf77 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_17.run(buf76, buf77, 128, 392, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf78 = aten.convolution_backward(buf75, relu_2, primals_7, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_7
        buf79 = buf78[0]
        buf80 = buf78[1]
        del buf78
        buf81 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_18.run(relu_2, buf79, buf81, 50176, 128, grid=grid(50176, 128), stream=stream0)
        del buf79
        del relu_2
        buf82 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_16.run(buf81, buf82, 50176, 128, grid=grid(50176), stream=stream0)
        buf83 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_17.run(buf82, buf83, 128, 392, grid=grid(128), stream=stream0)
        del buf82
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf84 = aten.convolution_backward(buf81, getitem, primals_5, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf81
        del getitem
        del primals_5
        buf85 = buf84[0]
        buf86 = buf84[1]
        del buf84
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf87 = aten.max_pool2d_with_indices_backward(buf85, relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_1)
        del buf85
        del getitem_1
        buf88 = buf87
        del buf87
        buf89 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_19.run(buf89, relu_1, 12845056, grid=grid(12845056), stream=stream0)
        del relu_1
        buf90 = empty_strided((64, 1024), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_20.run(buf89, buf90, 65536, 196, grid=grid(65536), stream=stream0)
        buf91 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_21.run(buf90, buf91, 64, 1024, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf92 = aten.convolution_backward(buf89, relu, primals_3, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_3
        buf93 = buf92[0]
        buf94 = buf92[1]
        del buf92
        buf95 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_22.run(relu, buf93, buf95, 200704, 64, grid=grid(200704, 64), stream=stream0)
        del buf93
        del relu
        buf96 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_20.run(buf95, buf96, 65536, 196, grid=grid(65536), stream=stream0)
        buf97 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_21.run(buf96, buf97, 64, 1024, grid=grid(64), stream=stream0)
        del buf96
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf98 = aten.convolution_backward(buf95, primals_33, primals_1, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf95
        del primals_1
        del primals_33
        buf99 = buf98[1]
        return (buf99, buf97, buf94, buf91, buf86, buf83, buf80, buf77, buf72, buf69, buf66, buf63, buf60, buf57, buf52, buf49, buf46, buf43, buf40, buf37, buf32, buf29, buf26, buf23, buf20, buf17, reinterpret_tensor(buf9, (4096, 25088), (25088, 1), 0), buf10, reinterpret_tensor(buf5, (4096, 4096), (4096, 1), 0), buf6, reinterpret_tensor(buf1, (1000, 4096), (4096, 1), 0), buf2, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((4, 64, 224, 224), (3211264, 1, 14336, 64), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((4, 64, 224, 224), (3211264, 1, 14336, 64), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.int64)
    relu_2 = rand_strided((4, 128, 112, 112), (1605632, 1, 14336, 128), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((4, 128, 112, 112), (1605632, 1, 14336, 128), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.int64)
    relu_4 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.int64)
    relu_7 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.int64)
    relu_10 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    getitem_8 = rand_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.int64)
    view = rand_strided((4, 25088), (25088, 1), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    clone_1 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((1000, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.bool)
    permute_7 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.bool)
    permute_11 = rand_strided((4096, 25088), (25088, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_33, relu, relu_1, getitem, getitem_1, relu_2, relu_3, getitem_2, getitem_3, relu_4, relu_5, relu_6, getitem_4, getitem_5, relu_7, relu_8, relu_9, getitem_6, getitem_7, relu_10, relu_11, relu_12, getitem_8, getitem_9, view, clone, clone_1, permute_3, le, permute_7, le_1, permute_11, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vgg16', benchmark_compiled_module)
