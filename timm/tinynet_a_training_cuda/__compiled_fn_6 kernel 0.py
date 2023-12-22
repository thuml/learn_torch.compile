
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


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxtta5zuyark6jtsop7tbmsf4zsqotzberqjau57atylxyc2owl.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_native_batch_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1280*((r2 + (96*x1)) // 36))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (1280*r2) + (122880*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (1280*r2) + (122880*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 36.0
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


# kernel path: /tmp/torchinductor_youkaichao/ko/ckogjzcilbgzhpcdcxn4hr4zgql4ruqnhfovm5gnmzh3tlm3r3mk.py
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
    rnumel = 3
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


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhkukqekursiej5cmrsbte3a7xkqyq2bnkfxvzlpr5lacc7cmd6.py
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
    rnumel = 3
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


# kernel path: /tmp/torchinductor_youkaichao/na/cnaavim3y4rcizfqbgl6ak4haa5s6czigrhxa6xnh7ihpyz2t5ps.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 1280
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = (yindex // 36)
    y3 = yindex
    y0 = yindex % 36
    tmp0 = tl.load(in_ptr0 + (x2 + (1280*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 36.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.003472222222222222
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (y0 + (36*x2) + (46080*y1)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wu/cwugt6piwxmc2hedggca337egjx6eaay7ymbwfs7hbobp4bakolo.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 320
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 36
    r2 = (rindex // 36)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (11520*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pl/cpl2d7z6yfxgspoetgpznduovrivo33cwdofzz75mgqxnbaga2bf.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 96
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
        tmp0 = tl.load(in_ptr0 + ((6*(((r2 + (96*x1)) // 6) % 6)) + (36*x0) + (11520*((r2 + (96*x1)) // 36)) + (r2 % 6)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (320*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmisznl5bqgba4jsitnet2wqcylshl5kwyunlyyaizp3ncipt3fh.py
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
    rnumel = 3
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


# kernel path: /tmp/torchinductor_youkaichao/5d/c5decozw3kqkblgrb33dbishcm45qjekhzkimvfwf6ebwkqidein.py
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
    xnumel = 36
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (320*x2) + (11520*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.003472222222222222
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
    tl.store(in_out_ptr0 + (x2 + (36*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gm/cgmc6bsyuvq5324ffmvlraehm45urfjtkmjb5etlf7j5aqwl7ur5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_309], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___6_____0___se_gate => sigmoid_75
# x_309 => mul_465, sigmoid_73
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
    rnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (r2 + (36*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1152*r2) + (41472*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3q/c3q3aqtxxxuznl77yf6w5yxewp2hphgka3hnk3wxdnxp47d7rzyb.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___6_____0___se_gate => sigmoid_75
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3456
    rnumel = 96
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
        tmp0 = tl.load(in_ptr0 + ((6*(((r2 + (96*x1)) // 6) % 6)) + (36*x0) + (41472*((r2 + (96*x1)) // 36)) + (r2 % 6)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1152*((r2 + (96*x1)) // 36))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (1152*((r2 + (96*x1)) // 36))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (1152*r2) + (110592*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (1152*r2) + (110592*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 36.0
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


# kernel path: /tmp/torchinductor_youkaichao/63/c63dldnhxiusxmg33krzm56aiasobxejxr3sysc5btt5bk5iloyq.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___6_____0___se_gate => sigmoid_75
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
    rnumel = 3
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


# kernel path: /tmp/torchinductor_youkaichao/o5/co5s3ztxds75vpbbjk6jcdwzk5fvn72mqmgulaxiy754a5w375w4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___6_____0___se_gate => sigmoid_75
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
    rnumel = 3
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


# kernel path: /tmp/torchinductor_youkaichao/ec/cec5zy4z2vdgylcuhuafl2col35fgbvkvvyufk5klajwibq6gobl.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___6_____0___se_gate => sigmoid_75
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 1152
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 36
    y1 = (yindex // 36)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (36*x2) + (41472*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp5 = 36.0
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
    tmp20 = 0.003472222222222222
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (1152*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6b/c6bbatmhinztcpnkfqrk2qfxcbccrhhfh74t53cicsgg5ea5lkmo.py
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
    xnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (y0 + (1152*x2) + (41472*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (36*y3)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b7/cb7qdwqmbcddwtuhxt2cfy2ufvwjsuyhak3fsdaqik53gunpn4wr.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3456
    rnumel = 96
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
        tmp0 = tl.load(in_ptr0 + ((6*(((r2 + (96*x1)) // 6) % 6)) + (36*x0) + (41472*((r2 + (96*x1)) // 36)) + (r2 % 6)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1152*r2) + (110592*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (1152*r2) + (110592*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ij/cij57ylwj6522apathv6ll66g4obs6lci4uz6ugouups33ayq4yb.py
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
    xnumel = 36
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (36*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (1152*x2) + (41472*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1152*x2) + (41472*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.003472222222222222
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
    tl.store(in_out_ptr0 + (x2 + (36*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s6/cs6uwbfl5nehnazvskjec5tv5gop3pfh7w4nmvesamat4h4752ep.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 192
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 36
    r2 = (rindex // 36)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (6912*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gm/cgmkw5wzymnwaqgoaak5cdowzz3mu77ewbygjza4xifn4ve7bagr.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 96
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
        tmp0 = tl.load(in_ptr0 + ((6*(((r2 + (96*x1)) // 6) % 6)) + (36*x0) + (6912*((r2 + (96*x1)) // 36)) + (r2 % 6)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (192*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vn/cvnzorbevoiiek74ec4tk5kot2pysx6ddj6u644dcd23jppeewtb.py
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
    rnumel = 3
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


# kernel path: /tmp/torchinductor_youkaichao/he/che3l7fv2shfvybusjkaqjjecfgzlf6hts7zbadxysvmxhz6hbnq.py
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
    xnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (192*x2) + (6912*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.003472222222222222
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (36*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wa/cwaew33omwprbgdwnq7n47zhzdyuxgpjvz6pgcnsmdoulja6xouc.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 192
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 36
    r2 = (rindex // 36)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (6912*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (36*x0) + (6912*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ph/cph375z3qn5efizx2catcxiqvvwmbzsambfg4h46dyyckd4463xh.py
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
    xnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (192*x2) + (6912*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.003472222222222222
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2 + (36*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xz/cxz2bvu55yxcrrbpdrglulw6ixahskzwhhourj574tixk3dw43sa.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 192
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 36
    r2 = (rindex // 36)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (6912*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (36*x0) + (6912*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (36*x0) + (6912*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/s4/cs43a6pxzjjz66fcfucyudlo4c4v2q4q4x7fe32rkcqe6f2dawzl.py
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
    xnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (192*x2) + (6912*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.003472222222222222
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (36*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/od/code7mrfis5i332lf6vkck2cckxguymsmiproawhvzq5oaylqp64.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 192
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 36
    r2 = (rindex // 36)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (6912*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (36*x0) + (6912*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (36*x0) + (6912*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (36*x0) + (6912*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vp/cvpj5ealvc2afagn4syjohafj4jyo5bhhg3kyno3sgnjslumfus5.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 + (192*x2) + (6912*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 - tmp8
    tmp11 = 0.003472222222222222
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
    tl.store(in_out_ptr0 + (x2 + (36*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdrch5eb7lvxkehzfbjcylqr4icwrjacthxawflyh3bsk6ghprhu.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndgwkcih4rtne273qbt64j4l6y4utcybogegty4fuhrlhewn5jz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 36
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (192*x2) + (6912*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.003472222222222222
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
    tl.store(in_out_ptr0 + (x2 + (36*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cico64kdega7pvpuovolkn6m2g2sxfbgdjkww3yp5ryjvabunxte.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_225], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_55
# x_225 => mul_340, sigmoid_53
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_32', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (r2 + (36*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (672*r2) + (24192*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqq3kqauaar5l3kg26w5l2wz4n246zzidv6kxlgy33urijq32c2.py
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
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_33', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ej/cejgyt3kebl5pb5cbevzvwg7nieudumaxflg3t7cy3aqzulmxorz.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_34', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/2y/c2ytn23alnrw24upidjb6wwjoftzbkhpfdi7ohqv6clzdapj4nba.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_35', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zr/czrkftuyljg5izjuicq6aomn5p57uuirlsbtejuyxs2icpr7s2xb.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_55
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2016
    rnumel = 96
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
        tmp0 = tl.load(in_ptr0 + ((6*(((r2 + (96*x1)) // 6) % 6)) + (36*x0) + (24192*((r2 + (96*x1)) // 36)) + (r2 % 6)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (672*((r2 + (96*x1)) // 36))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (672*((r2 + (96*x1)) // 36))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (672*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (672*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 36.0
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


# kernel path: /tmp/torchinductor_youkaichao/md/cmdi7crnyfkzbz26efonknqwjxkyvcvrz7zj3ptoeme4j7nquaz6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_55
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 3
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


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfv6wxc4ezfe4i3wape2orxtoiuejhapgrrdb2d76g5xuj4yffa.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_55
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 3
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


# kernel path: /tmp/torchinductor_youkaichao/sm/csmqwuk4q7c5k6ebwvbfvz3qk2eym2thdfr4prct565kvfurl2wm.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_55
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 672
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 36
    y1 = (yindex // 36)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (36*x2) + (24192*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp5 = 36.0
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
    tmp20 = 0.003472222222222222
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (672*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sl/cslvd6bugf2jl5eigdp3hdt3pzf2iq5tzp2e4qffw3kcnxux4zwd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (y0 + (672*x2) + (24192*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (36*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tv/ctvz5jap3ptp732j5zz7uhlujgea5njbvmfhsgowkib76vykkrik.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x1) + (96768*((r2 + (128*x0)) // 144)) + ((r2 + (128*x0)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (672*r2) + (86016*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpspaeccrjqj7kurgdcvq3gnscr5uff3s5gjv4q7l43qd65x4u3z.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xy/cxymptwx2lhkzf6fntwaxzqycbpegog4vitfmqjp7kr2oguynts2.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x0) + (96768*((r2 + (128*x1)) // 144)) + ((r2 + (128*x1)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (672*r2) + (86016*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (672*r2) + (86016*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xu/cxu6gfst5nit4twji67ejd3wfdm4kanxmzbiqesxv3giff77vnyt.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 9
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


# kernel path: /tmp/torchinductor_youkaichao/om/com2spsops4wpg5zkhdc67hbwnpuvcuyfnz52m7hc6erqdztozcm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 144
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (672*x2) + (96768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (672*x2) + (96768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0008680555555555555
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
    tl.store(in_out_ptr0 + (x2 + (144*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qm/cqmbmh2oqb6syf5invlnrulb6o4hzzfxdmt3obuin7twidcfta3i.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 144
        r2 = (rindex // 144)
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0) + (16128*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexel2ku64gaxhmg3vq4dreowgkyqkkgu35he5k2rziubreueg26.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1008
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x1) + (16128*((r2 + (128*x0)) // 144)) + ((r2 + (128*x0)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (112*r2) + (14336*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ci/cciwanizbkn4wsmsp3zzvbq3fiw55yrxcm7i2wzs2jpkgsghrxgr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dfleutyrtujf3y4qpwzm323gapold3qu4etvmrbiffwvpttnug.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (112*x2) + (16128*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0008680555555555555
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rd/crdg4d6g2filbu6lkadr6a5nn3sazeozvxoj4lxnxxpp7pua4c4s.py
# Source Nodes: [x_208], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_208 => mul_315, sigmoid_49
triton_red_fused_mul_silu_sum_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (r3 + (72*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (672*r3) + (48384*x0) + (96768*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47ovuqgqyvl2zotmhtfrusqfxdor5koriadlyqeavbsb77lfu4z.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_208], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___4_____3___se_gate => sigmoid_51
# x_208 => mul_315, sigmoid_49
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_51', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5fzawmuhoih42iortne7bh6ez2fg5bk6ptjz4epgbnonxkattl.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____3___se_gate => sigmoid_51
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x1) + (96768*((r2 + (128*x0)) // 144)) + ((r2 + (128*x0)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (672*((r2 + (128*x0)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (672*((r2 + (128*x0)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (672*r2) + (86016*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 144.0
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


# kernel path: /tmp/torchinductor_youkaichao/lu/clusoxgmqvmklhnwvz6hfe35eeh4h6gfo6rpjhmnzk3fyzo4reua.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____3___se_gate => sigmoid_51
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x0) + (96768*((r2 + (128*x1)) // 144)) + ((r2 + (128*x1)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (672*((r2 + (128*x1)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (672*((r2 + (128*x1)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (672*r2) + (86016*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (672*r2) + (86016*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 144.0
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


# kernel path: /tmp/torchinductor_youkaichao/5k/c5kinsfypn4axzickf2mtnqnozq46jacya7pvvkvk5ynl6nxofvn.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____3___se_gate => sigmoid_51
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 672
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
    tmp0 = tl.load(in_ptr0 + (y0 + (144*x2) + (96768*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp5 = 144.0
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
    tmp20 = 0.0008680555555555555
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (672*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ll/cllabnvhdsxxxmlxtyjkwwocf5tqepyjbxj7hxywmmru4qjrmbat.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (y0 + (672*x2) + (96768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dsj4ihnfp4e7tnxqy6qis66rtkid5ev2yyxmkz6qvylr4yetdd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 1152
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
        r1 = rindex % 144
        r2 = (rindex // 144)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0) + (16128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (144*x0) + (16128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tb/ctbqqhrnzo4dfvljkgjzy547ovkcdhacz36ojaeilenbx6tw7vc4.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (112*x2) + (16128*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0008680555555555555
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/md/cmdayaprfuefpkwmy7fd7ktkap2pyvqg744ju3w4iey5slschtq2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 1152
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
        r1 = rindex % 144
        r2 = (rindex // 144)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0) + (16128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (144*x0) + (16128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (144*x0) + (16128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ym/cymnh7a25jtcylu25n5yvobkomuczezv5vxmaaf5y64up6qvck2i.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (112*x2) + (16128*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0008680555555555555
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/im/cimh74qjv3cm7sxodkgnhuxcmaxy7rcjpjz6wmibc5rrerob4ayn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 144
        r2 = (rindex // 144)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0) + (16128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (144*x0) + (16128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (144*x0) + (16128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (144*x0) + (16128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (112*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp17 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp15 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqwigrvq6eawx2qtbye5dtds3gjntyficwlher4sxbcokb5ymhd.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 144
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (112*x2) + (16128*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0008680555555555555
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
    tl.store(in_out_ptr0 + (x2 + (144*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tb/ctbieqgnujomnrh7sz2vtorwxnpp4tx6edpf6trdudhb5an2bzdb.py
# Source Nodes: [x_158], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_158 => mul_240, sigmoid_37
triton_red_fused_mul_silu_sum_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (r3 + (72*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (480*r3) + (34560*x0) + (69120*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wot5drhkdoliykacgmrvne2j4wbilefco7kctcwmn6hbauwnq5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_158], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_39
# x_158 => mul_240, sigmoid_37
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_63', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjvkxp6v25zzfc7kuul5jgmae5sq4ls2hz5bxt7okiadiunn5x2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_64', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/eq/ceqj5wrx4cg3kdrzscdet2ehn6ev6gbckb4avbcqrmxbebgtir7z.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_65', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/7a/c7ahysmemng3feo62bpgtwzmfytpvomkoib3lxzw3feorywd33to.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_66', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qf/cqfarwvg24k3hqsbrd2kn43zphzcdm4irur7w3p6t5s5txxcgcfj.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_39
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4320
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x1) + (69120*((r2 + (128*x0)) // 144)) + ((r2 + (128*x0)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (480*((r2 + (128*x0)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (480*((r2 + (128*x0)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (480*r2) + (61440*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 144.0
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


# kernel path: /tmp/torchinductor_youkaichao/45/c45sgkoqtmw44kmi54gojxbp5uxuo52q76xg5xgn57d7ssyaevef.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_39
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y3/cy35p6mbng7m2evsbcqqlozmtvk5gkzhcdjwjm2jstku2nr433jh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_39
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4320
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x0) + (69120*((r2 + (128*x1)) // 144)) + ((r2 + (128*x1)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (480*((r2 + (128*x1)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (480*((r2 + (128*x1)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (480*r2) + (61440*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (480*r2) + (61440*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 144.0
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


# kernel path: /tmp/torchinductor_youkaichao/hy/chy6rsoiotwceoqiret6tmuwjrgprhe6l4djyf7tbbzptzhv4ofo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_39
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 9
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


# kernel path: /tmp/torchinductor_youkaichao/ng/cng7mn624yg4wu4veyzwrmqme4sbc75ep2nwq5g75idesmgwenr3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_39
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 480
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
    tmp0 = tl.load(in_ptr0 + (y0 + (144*x2) + (69120*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp5 = 144.0
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
    tmp20 = 0.0008680555555555555
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (480*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbqyefx5xiwcjg3jqnov7mnylrb4jlej2kz7ro3aaeklvy2rrh3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (y0 + (480*x2) + (69120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7wwgmhydpsj6jomufuu5wqth4rughmb3shsyt4wmlurxr7kpwg.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4320
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x1) + (69120*((r2 + (128*x0)) // 144)) + ((r2 + (128*x0)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (480*r2) + (61440*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2rbwtsalyq4wjsyrsd3k7cauoxbqnkr75mj7tsykvuhgims3pj.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4320
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x0) + (69120*((r2 + (128*x1)) // 144)) + ((r2 + (128*x1)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (480*r2) + (61440*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (480*r2) + (61440*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmz435ltjougkck2w3axle75744svklq3cpchwvh5oagcdm2vkva.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_75', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 144
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (480*x2) + (69120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (480*x2) + (69120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0008680555555555555
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
    tl.store(in_out_ptr0 + (x2 + (144*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c2/cc26rkrdlsdk5kzphtwnkclkgtqb7xlmarfb7jqip4hudepqmmlq.py
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
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 144
        r2 = (rindex // 144)
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0) + (11520*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/clidbay2vr37uxx44dq3qmpjluiwkgdycvx46rcqjto6mqoeyr7g.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 720
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x1) + (11520*((r2 + (128*x0)) // 144)) + ((r2 + (128*x0)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (80*r2) + (10240*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n4/cn4o3bdkaz77h6iuh4ebb6zz4kizcwc3blflx2xbkievyhtdtqsi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c25sdx6wljajswhoeysyhzp5ppzkmru7ymyaiki2qbtovysvg3vk.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (80*x2) + (11520*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0008680555555555555
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ga/cgaqitolkf4kn7i7nepl5h3ppvk2hw6xgmqbulqpvcqstktmi5eq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 1152
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
        r1 = rindex % 144
        r2 = (rindex // 144)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0) + (11520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (144*x0) + (11520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zm/czmxtzn4mzx4o7hhoclcbnefctfcc2pyygfotly57xqibdtawxs7.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (80*x2) + (11520*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0008680555555555555
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4unpuap7uxpgg3hxnou4j334opjtmarsggiogvzq2jvmvegtr7x.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 1152
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
        r1 = rindex % 144
        r2 = (rindex // 144)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0) + (11520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (144*x0) + (11520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (144*x0) + (11520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/go/cgo2pqommfnjq6lwnyid2dmda4pias2bbxldhsqypmevzcpqljmu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (80*x2) + (11520*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0008680555555555555
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4zn4kdaluw5jgjprkwaylowxzynvvuo6dr6usk2jlklkgsbqry.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 144
        r2 = (rindex // 144)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0) + (11520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (144*x0) + (11520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (144*x0) + (11520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (144*x0) + (11520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (80*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp17 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp15 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csb3ayhwbtxll2ya66ejeo6r76zlrmjjqjdtnkwvn7fu6yuzyoij.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_85', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 144
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (80*x2) + (11520*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0008680555555555555
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
    tl.store(in_out_ptr0 + (x2 + (144*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqh3cp3ocxdzgjfynq2ybknonksxtv7hywjtklis7xzmnayh7iki.py
# Source Nodes: [x_91], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_91 => mul_140, sigmoid_21
triton_red_fused_mul_silu_sum_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (r3 + (72*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*r3) + (17280*x0) + (34560*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4cgys6skiisvc25mfinnixhpxjfb2r24slxsnlarl6zwawwkuf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
# x_91 => mul_140, sigmoid_21
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_87', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/2t/c2tpiaagh73tziyt5ezo5aotwovy6pwcqtr64y4ejrocjo7qgebh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_88', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/bu/cbutwmqqxib5pyfgir6k7zh7qmyjjari32qht6gz6nzrf4gabemd.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_89', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/b5/cb56vykzszdmbrgncjasyfqevhcrwfr52b2l64fxfbfxursdyew7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_90', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/26/c26gyrx26ttnqkajhmajjkvz72httnxt3cj4vrp2siwzrcfv3ptk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2160
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((144*x1) + (34560*((r2 + (128*x0)) // 144)) + ((r2 + (128*x0)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*((r2 + (128*x0)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (240*((r2 + (128*x0)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (240*r2) + (30720*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 144.0
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


# kernel path: /tmp/torchinductor_youkaichao/zv/czvkt2ue7fvc6kzabq63ehb55ih6lxxl2c2kldx6dzbc4ltvqdvr.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xe/cxezouppqlm5mrqjjwlrydprd3cj5vtvfxqk2exm7gi7zfhqkpuv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2160
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
        tmp0 = tl.load(in_ptr0 + ((144*x0) + (34560*((r2 + (128*x1)) // 144)) + ((r2 + (128*x1)) % 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (240*((r2 + (128*x1)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (240*((r2 + (128*x1)) // 144))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 144.0
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


# kernel path: /tmp/torchinductor_youkaichao/nn/cnnvral5wdif6234zwkg3qnakthalzjenwlxewayezyptlsqzmue.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 9
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


# kernel path: /tmp/torchinductor_youkaichao/r3/cr34wrndx7e2rfj6v7kv6lya2b4vzitti5kkow7efhg2vhlrac2n.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 240
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
    tmp0 = tl.load(in_ptr0 + (y0 + (144*x2) + (34560*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp5 = 144.0
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
    tmp20 = 0.0008680555555555555
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpspksfvyg2jteg5ueaeap2minxybdqqk4kbdye7e4zxyc5txufi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (y0 + (240*x2) + (34560*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/ckishkg5znutnhtjlnvawzagxpty5gqjyyl4k6cwcpeswt4pok6x.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8640
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 36
    x1 = (xindex // 36)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((576*x1) + (138240*((r2 + (128*x0)) // 576)) + ((r2 + (128*x0)) % 576)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*r2) + (30720*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3s/c3sritgcd5myxcpul5bl6b2nk62lgom42q3cxzc5dbnn2cs4xnrg.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4bgfhgx732bgo357opke3iqewk3trobj7xxpehmuux4ml3g5fg.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8640
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
        tmp0 = tl.load(in_ptr0 + ((576*x0) + (138240*((r2 + (128*x1)) // 576)) + ((r2 + (128*x1)) % 576)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rd/crdjftqha7jxwtp7ri34shtvgs4bdfjztudam354ac5kwtlldvir.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 36
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


# kernel path: /tmp/torchinductor_youkaichao/sy/csy4aqzzxwvjffohrql32nfm4zkqwnnt4yyg5gcoyenc3zrajepp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_101', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 576
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (240*x2) + (138240*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (240*x2) + (138240*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00021701388888888888
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
    tl.store(in_out_ptr0 + (x2 + (576*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54xwimcus5zb3sk3icc74sie3rsn4qwc5jdphpmckl22ueo6xsk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 576
        r2 = (rindex // 576)
        tmp0 = tl.load(in_ptr0 + (r1 + (576*x0) + (23040*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r7/cr7fr5q5bkmagpzgib7tz3b5jgp4ignqapakaepd6v4d7cx3uxdg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1440
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 36
    x1 = (xindex // 36)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((576*x1) + (23040*((r2 + (128*x0)) // 576)) + ((r2 + (128*x0)) % 576)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (40*r2) + (5120*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhdcjibacjheyxeofsrrdjvare5v5c642oydj4zsupdp7tck24b.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfayarsjz6loyqk3usnwfter5rqkra5wfilguawfu4lvrn6wdpwr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_105', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 576
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
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (40*x2) + (23040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00021701388888888888
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (576*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/chedkpuo53jfkfhdtpemj376q44bn52noa5b6bgpuvuvvo4v5opz.py
# Source Nodes: [x_74], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_74 => mul_115, sigmoid_17
triton_red_fused_mul_silu_sum_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9600
    rnumel = 116
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x4 = (xindex // 5)
    x1 = (xindex // 5) % 240
    x2 = (xindex // 1200)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (116*x0)
        tmp1 = tl.full([1, 1], 576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((576*x4) + ((r3 + (116*x0)) % 576)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (240*((r3 + (116*x0)) % 576)) + (138240*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rl/crlljkb6t7d3h3h7wth4xnwpetdcvi7atakriilbwgijoqnx3de5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
# x_74 => mul_115, sigmoid_17
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_107', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (5*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ta/ctaj5z2hptwwc4fol7kx6akkuqe7bvif2nchush5ygm34wptpvy6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8640
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 36
    x1 = (xindex // 36)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((576*x1) + (138240*((r2 + (128*x0)) // 576)) + ((r2 + (128*x0)) % 576)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*((r2 + (128*x0)) // 576))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (240*((r2 + (128*x0)) // 576))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (240*r2) + (30720*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 576.0
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


# kernel path: /tmp/torchinductor_youkaichao/oe/coeenov2aazbmrjvu23cu5cdbena5nbyq7h7idfegwlgq2juzr6c.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_109', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8640
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
        tmp0 = tl.load(in_ptr0 + ((576*x0) + (138240*((r2 + (128*x1)) // 576)) + ((r2 + (128*x1)) % 576)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (240*((r2 + (128*x1)) // 576))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (240*((r2 + (128*x1)) // 576))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 576.0
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


# kernel path: /tmp/torchinductor_youkaichao/kr/ckrhq6pnd2xmvleuwlqv6fyym45nprzjmotr4odzahov5lmj6xaa.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_110', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
    xnumel = 240
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 576
    y1 = (yindex // 576)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (576*x2) + (138240*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp5 = 576.0
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
    tmp20 = 0.00021701388888888888
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccruoen24ryh6ub2soy3bw2igjzqndvpze4wq5be25yis44ugdoq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_111', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 576
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
    tmp0 = tl.load(in_ptr0 + (y0 + (240*x2) + (138240*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (576*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/cclfalwn4cch7akotcrkbeytr6st6czgmxjebmenhkkd6dztxvgh.py
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
    rnumel = 4608
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
        r1 = rindex % 576
        r2 = (rindex // 576)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (576*x0) + (23040*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (576*x0) + (23040*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7u3uiehwyhz75z3irctgmrgis5vgco6ls5mamwcgk6rodza46r.py
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
    xnumel = 576
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (40*x2) + (23040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00021701388888888888
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
    tl.store(in_out_ptr0 + (x2 + (576*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lm/clmed2h3g4q3tvms6jaf37r3wqm6iynyexwku6vvnbazz7b5f2mh.py
# Source Nodes: [x_58], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_58 => mul_90, sigmoid_13
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5760
    rnumel = 116
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x4 = (xindex // 5)
    x1 = (xindex // 5) % 144
    x2 = (xindex // 720)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (116*x0)
        tmp1 = tl.full([1, 1], 576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((576*x4) + ((r3 + (116*x0)) % 576)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (144*((r3 + (116*x0)) % 576)) + (82944*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5venxctuhds22om632ekbgyxagdbjxyaqkhtbxqr2lf24har6x.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
# x_58 => mul_90, sigmoid_13
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
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (5*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/df/cdfml677wkmi55qqjjpm5nkc7lwo3hk2d7agfh4y2kojv5r4375l.py
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
    xnumel = 5184
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 36
    x1 = (xindex // 36)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((576*x1) + (82944*((r2 + (128*x0)) // 576)) + ((r2 + (128*x0)) % 576)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*((r2 + (128*x0)) // 576))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (144*((r2 + (128*x0)) // 576))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (144*r2) + (18432*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 576.0
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


# kernel path: /tmp/torchinductor_youkaichao/sq/csqaxzg4rrkdgb6yeszhgzy5mvwgmtmvjlbhcvrmyk7i74evippe.py
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
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chxnr3o63b3my42aequysghdkr7y7hrismsyvrgbh53mf6sqaeaw.py
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
    xnumel = 5184
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
        tmp0 = tl.load(in_ptr0 + ((576*x0) + (82944*((r2 + (128*x1)) // 576)) + ((r2 + (128*x1)) % 576)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (144*((r2 + (128*x1)) // 576))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (144*((r2 + (128*x1)) // 576))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 576.0
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


# kernel path: /tmp/torchinductor_youkaichao/ko/ckolqyranregf7ry3hhc3ovnicdzfb7zixe437pdbomy2gkdcjzm.py
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
    rnumel = 36
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


# kernel path: /tmp/torchinductor_youkaichao/2g/c2gv4bvuvaw2mriljg3nxj65a4mscc24qi7jvhdydmn345kf72nr.py
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
    ynumel = 4608
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 576
    y1 = (yindex // 576)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (576*x2) + (82944*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp5 = 576.0
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
    tmp20 = 0.00021701388888888888
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/chew3ja4md5327c72r47mdeqx3vbaaaawuz5euoaf3hisgau6kwd.py
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
    xnumel = 576
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
    tmp0 = tl.load(in_ptr0 + (y0 + (144*x2) + (82944*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (576*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/cckmjcuhycmo7kxeoysmfixu34uc5xgthjcikhxn7fjegev37wjv.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_125', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20736
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2304*x1) + (331776*((r2 + (128*x0)) // 2304)) + ((r2 + (128*x0)) % 2304)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*r2) + (18432*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vw/cvwo326hby7aqqiwqqbioyssu6p22uvsjyiknfvsanxyz5mp6vld.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_126 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_126', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrlqunnjl57zktdiy2ltdkersrnewc6mmy74avl5yydbq3xipeb.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20736
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
        tmp0 = tl.load(in_ptr0 + ((2304*x0) + (331776*((r2 + (128*x1)) // 2304)) + ((r2 + (128*x1)) % 2304)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qp/cqpxmk4b3dizmjlvb3d3oln7ynfptwlizv4pyqvorplrploe5qqc.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_128 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_128', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 144
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


# kernel path: /tmp/torchinductor_youkaichao/lq/clqheruaek6cjmgd3x5araztggahhzeawmm2i5ygzirilcxnnknr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_129', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 2304
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (144*x2) + (331776*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (144*x2) + (331776*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 5.425347222222222e-05
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
    tl.store(in_out_ptr0 + (x2 + (2304*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tm/ctmuqqf5jfh45lgipq7gdd27kd52i527u2szfiny5bj7liw3h5p2.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 6144
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
        tmp0 = tl.load(in_ptr0 + ((48*(((r2 + (6144*x1)) // 48) % 48)) + (2304*x0) + (55296*((r2 + (6144*x1)) // 2304)) + (r2 % 48)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nn/cnncy5tu3djromxob2nrq4qnr5rbprg2on5bppuhbhczwaawtaei.py
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
    rnumel = 3
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


# kernel path: /tmp/torchinductor_youkaichao/tv/ctvmzlu4vh7v2mvk66mm7jm5kcoqorouj57solsi3qbwl4nb33st.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_132', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3456
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2304*x1) + (55296*((r2 + (128*x0)) // 2304)) + ((r2 + (128*x0)) % 2304)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (24*r2) + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqazy2tbu5lk2jnqppfi7nrjsuluuv3hd4s52rivvtjur46fpnzm.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_133', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r6/cr6dvmum6o3kfnioa2wqnj6xt3eo6xpg4npcw6i2bmt6hcyaodxk.py
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
    xnumel = 2304
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
    tmp0 = tl.load(in_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (24*x2) + (55296*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 5.425347222222222e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (2304*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxozrlawpcijgf2ylleniqexac7obsg5cmjvulneuurpwevrqtn.py
# Source Nodes: [x_41], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_41 => mul_65, sigmoid_9
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_135', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20736
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 18
    x1 = (xindex // 18) % 144
    x2 = (xindex // 2592)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*r3) + (18432*x0) + (331776*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sg/csg4qtrf7i7phc4k6utvwrunykgcgwhr3dapriqwjvj672eaxruy.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
# x_41 => mul_65, sigmoid_9
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
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (18*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4f/c4fslqhbmpkgwkvq6wpxxr4uxrri2jagtfgcs5u7bbffavf56ftu.py
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
    xnumel = 20736
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2304*x1) + (331776*((r2 + (128*x0)) // 2304)) + ((r2 + (128*x0)) % 2304)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*((r2 + (128*x0)) // 2304))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (144*((r2 + (128*x0)) // 2304))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (144*r2) + (18432*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 2304.0
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


# kernel path: /tmp/torchinductor_youkaichao/yl/cylmc5tafjallmurm4qjunyuuxn6ukrjwkulfprghfpbef2bglkf.py
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
    xnumel = 20736
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
        tmp0 = tl.load(in_ptr0 + ((2304*x0) + (331776*((r2 + (128*x1)) // 2304)) + ((r2 + (128*x1)) % 2304)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (144*((r2 + (128*x1)) // 2304))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (144*((r2 + (128*x1)) // 2304))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 2304.0
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


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5j4va52qz6v3w6dbegvpqywxittco6g2zamv23f6x3i7lrxmbw.py
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
    ynumel = 18432
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2304
    y1 = (yindex // 2304)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (2304*x2) + (331776*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (144*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (144*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (144*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (144*y3)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 2304.0
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
    tmp20 = 5.425347222222222e-05
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3an6befdbbzxbxeitpnax7ldlliu53rqqvv6ytx6selbehrckhf.py
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
    xnumel = 2304
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
    tmp0 = tl.load(in_ptr0 + (y0 + (144*x2) + (331776*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (2304*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cjenduu6jyoumgwmbyc24i6izzpt5frz45ep7axehzy22seobsff.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_141 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_141', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 6144
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
        tmp0 = tl.load(in_ptr0 + ((48*(((r2 + (6144*x1)) // 48) % 48)) + (2304*x0) + (55296*((r2 + (6144*x1)) // 2304)) + (r2 % 48)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((48*(((r2 + (6144*x1)) // 48) % 48)) + (2304*x0) + (55296*((r2 + (6144*x1)) // 2304)) + (r2 % 48)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (24*r2) + (147456*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jq35siuiuiztd6z24xyak7fzezxuvb67cn45bxywgdufko3z7c.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_142 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_142', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 3
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


# kernel path: /tmp/torchinductor_youkaichao/7p/c7pv342vy4mt3krczbqljy2nx6clr4v7t6v6cb5mhdt6lf6lvdpz.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_143 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_143', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 2304
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (24*x2) + (55296*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 5.425347222222222e-05
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
    tl.store(in_out_ptr0 + (x2 + (2304*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlth7dzt4mllr3q4vcmofgwuwdgcapzfz2tfr2mjcrj4sl7jbu6.py
# Source Nodes: [x_25], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_25 => mul_40, sigmoid_5
triton_red_fused_mul_silu_sum_144 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_144', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13824
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 18
    x1 = (xindex // 18) % 96
    x2 = (xindex // 1728)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*r3) + (12288*x0) + (221184*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3h/c3h5y6kav6ujebfbqmsvfuyxze3t7t2mex3frxrydcuqovw3fxeq.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
# x_25 => mul_40, sigmoid_5
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_145 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_145', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (18*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ci/cciyazvw2nf6xszwrrizqphwfio2oyf3twlojwspqctxyo2wyez3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_146 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_146', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nvf4fojyc4nb7ie2gujdafj7frux3rmdu3rgoirjvkxkeixnzy.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_147 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_147', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/bi/cbimtrgz2utq5gfezcmjsxs6ciokqd4552dxpbyddukv4dp6yyhq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_148 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_148', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlxslpc4wybflew6wdgto62w6kf7dqw7u5zwn5eujomvq2ol6u3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_149 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_149', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13824
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2304*x1) + (221184*((r2 + (128*x0)) // 2304)) + ((r2 + (128*x0)) % 2304)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*((r2 + (128*x0)) // 2304))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (96*((r2 + (128*x0)) // 2304))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 2304.0
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


# kernel path: /tmp/torchinductor_youkaichao/jl/cjlbzcl3qovg33q5x4wpyufls2lr73lkei6ku6fdwzsasxihd77d.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_150 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_150', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cncottbkpfun2jc3sazg3bsytq355po6hvpmes2bysws56llljzh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_151 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_151', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13824
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
        tmp0 = tl.load(in_ptr0 + ((2304*x0) + (221184*((r2 + (128*x1)) // 2304)) + ((r2 + (128*x1)) % 2304)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (96*((r2 + (128*x1)) // 2304))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (96*((r2 + (128*x1)) // 2304))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 2304.0
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


# kernel path: /tmp/torchinductor_youkaichao/3c/c3cb6yoeagapozzf6sgtpnylesuarl4tqrkhss5qk4dxrstwaf2u.py
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
    size_hints=[128, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_152', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 144
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


# kernel path: /tmp/torchinductor_youkaichao/i5/ci56hkwgvk6v5n5f24dycm7sywu3v3jovkc455glcjtgzuez24oy.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_153 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_153', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2304
    y1 = (yindex // 2304)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (2304*x2) + (221184*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (96*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (96*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (96*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (96*y3)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 2304.0
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
    tmp20 = 5.425347222222222e-05
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caqgpwpnzcittqiftybhk4ecra7r5o64y6fhvseupo3pkixqrmed.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_154 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_154', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 2304
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
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (221184*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (2304*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cicpwthf62l6bmuvowufgv7l5asbooh442axibgb4xkkpnx4mmkb.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_155 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_155', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 55296
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x1) + (884736*((r2 + (128*x0)) // 9216)) + ((r2 + (128*x0)) % 9216)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*r2) + (12288*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/if/cif2o4ikc4ebom42g2apcbw3hkqk5jkeqkbckosrph27fa2whddz.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_156 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_156', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 96
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxl6dak66jrfv76xodmndeuqxdernkqsw7bwwvbsobdxyewqd6ed.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_157 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_157', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 55296
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x0) + (884736*((r2 + (128*x1)) // 9216)) + ((r2 + (128*x1)) % 9216)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (96*r2) + (12288*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (96*r2) + (12288*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvk5v4juz7yih6joie7xcm474qa7csckcavgijil2fmqzebpzngj.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_158 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_158', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 576
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


# kernel path: /tmp/torchinductor_youkaichao/nl/cnl3elnruoenbno725obj6kpupkbukcq2e6l2sv7ojnr3kda3nu6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_159 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_159', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 9216
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (9216*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (96*x2) + (884736*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (96*x2) + (884736*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 1.3563368055555555e-05
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
    tl.store(in_out_ptr0 + (x2 + (9216*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ms/cmsuot7rjisw46iomhgujsfqevr5uzmzuz32abapjejnamtmnwtc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_160 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_160', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x1) + (147456*((r2 + (8192*x0)) // 9216)) + ((r2 + (8192*x0)) % 9216)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vr/cvrjtkijpyawrrvu56jwx5w3wwmo4a6toosrzqzgv7fvb6tj6d3v.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_161 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_161', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clqbmev36rq3ewtdfzqpugxlu3munl7pk7dyj367bpbgei7stapi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_162 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_162', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x1) + (147456*((r2 + (128*x0)) // 9216)) + ((r2 + (128*x0)) % 9216)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p3/cp37yy2ayte7xmmfy52rmqy7kaoosrxg7toruui5gzedxn55wbqa.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_163 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_163', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camvc2sub4qj7uk6nk3mgh6t3hxefooqaso255lqssqfemmwu2r7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_164 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_164', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 9216
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (9216*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (16*x2) + (147456*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1.3563368055555555e-05
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
    tl.store(in_out_ptr0 + (x2 + (9216*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpf3ucbffa3khuprpyabxfk6cglyk7q2tsjiphwhdriqspyqxy4q.py
# Source Nodes: [x_9], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_9 => mul_15, sigmoid_1
triton_red_fused_mul_silu_sum_165 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_165', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 72
    x1 = (xindex // 72) % 32
    x2 = (xindex // 2304)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r3) + (4096*x0) + (294912*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qd/cqdqc5vcynno2lhkpwv44qm2jt5vgw5j3th3uqlpduning4gnalu.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_9], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
# x_9 => mul_15, sigmoid_1
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_166 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_166', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 72
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (72*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/lq/clqwblbjxzx43iu4xn4kmzswbsxwf7knh53fh5dxe2vplr4ufake.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_167 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_167', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/7a/c7awqtyllmda54elgvr3p33nlzlf2k4ke2glvmaew3yngeqi6xqz.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_168 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_168', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6zkxoiuikxuho2bsbfh6ziejw6cgyolqoeud4bfzzhg6ou33hx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_169 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_169', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yb/cybrxt2hmhhittxuyxxmjv34khf3i2b55khiddcike52n4yn2knm.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_170 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_170', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x1) + (294912*((r2 + (128*x0)) // 9216)) + ((r2 + (128*x0)) % 9216)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*((r2 + (128*x0)) // 9216))), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (32*((r2 + (128*x0)) // 9216))), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (32*r2) + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 9216.0
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
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/no/cno4yvp2k4gbnppja33gabdc2xre7n5ng67fetl3eul2va22ovei.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_171 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_171', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 32
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7aw4tbbxeiibg6jf3tgbwdp4bqw6uzyw22xvmyn4jeelswq2m7.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_172 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_172', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x0) + (294912*((r2 + (128*x1)) // 9216)) + ((r2 + (128*x1)) % 9216)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (32*((r2 + (128*x1)) // 9216))), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (32*((r2 + (128*x1)) // 9216))), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 9216.0
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
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwygs6ysxvb25lian7jerz72dndigdta3f7s4qionhg26vth5fla.py
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
    size_hints=[32, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_173', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 576
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


# kernel path: /tmp/torchinductor_youkaichao/zz/czz7pwijudp4gzb5craf56htqtzk3e6do4dsovgvojkowvc6kqsz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_174 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_174', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 9216
    y1 = (yindex // 9216)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (9216*x2) + (294912*y1)), xmask, eviction_policy='evict_last')
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
    tmp5 = 9216.0
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
    tmp20 = 1.3563368055555555e-05
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgpzio262pwg5i5b2s2qczgaessdbtvpc3hlhiqzrcdtpkvbx72.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_175 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_175', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 9216
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
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x2) + (294912*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (9216*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/csevsivxflpnjuaia7lxmjnwkshj6xovcbsir4q2jccnbltlui34.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_176 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_176', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x1) + (294912*((r2 + (128*x0)) // 9216)) + ((r2 + (128*x0)) % 9216)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r2) + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n2/cn2dkibyh4rdbxlogrvbqhq6ecblaprmsahlelhz4gzpro7hqwzo.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_177 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_177', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x0) + (294912*((r2 + (128*x1)) // 9216)) + ((r2 + (128*x1)) % 9216)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xvhzs6zopdpqlcgoem5i3zf4g73vq4g5c2wwjafo7js54hir3l.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_178 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_178', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 9216
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (9216*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (32*x2) + (294912*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (32*x2) + (294912*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 1.3563368055555555e-05
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
    tl.store(in_out_ptr0 + (x2 + (9216*y3)), tmp19, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_118, primals_119, primals_121, primals_123, primals_124, primals_125, primals_126, primals_128, primals_130, primals_131, primals_132, primals_133, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_144, primals_145, primals_146, primals_147, primals_149, primals_151, primals_152, primals_153, primals_154, primals_156, primals_158, primals_159, primals_160, primals_161, primals_163, primals_165, primals_166, primals_167, primals_168, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_187, primals_188, primals_189, primals_191, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_233, primals_235, primals_236, primals_237, primals_238, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_249, primals_250, primals_427, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, add_9, mean, convolution_2, mul_16, convolution_3, mul_17, convolution_4, squeeze_7, add_14, convolution_5, squeeze_10, mul_32, convolution_6, squeeze_13, add_24, mean_1, convolution_7, mul_41, convolution_8, mul_42, convolution_9, squeeze_16, add_29, convolution_10, squeeze_19, mul_57, convolution_11, squeeze_22, add_39, mean_2, convolution_12, mul_66, convolution_13, mul_67, convolution_14, squeeze_25, add_45, convolution_15, squeeze_28, mul_82, convolution_16, squeeze_31, add_55, mean_3, convolution_17, mul_91, convolution_18, mul_92, convolution_19, squeeze_34, add_60, convolution_20, squeeze_37, mul_107, convolution_21, squeeze_40, add_70, mean_4, convolution_22, mul_116, convolution_23, mul_117, convolution_24, squeeze_43, add_76, convolution_25, squeeze_46, mul_132, convolution_26, squeeze_49, add_86, mean_5, convolution_27, mul_141, convolution_28, mul_142, convolution_29, squeeze_52, add_91, convolution_30, squeeze_55, mul_157, convolution_31, squeeze_58, add_101, mean_6, convolution_32, mul_166, convolution_33, mul_167, convolution_34, squeeze_61, add_107, convolution_35, squeeze_64, mul_182, convolution_36, squeeze_67, add_117, mean_7, convolution_37, mul_191, convolution_38, mul_192, convolution_39, squeeze_70, add_123, convolution_40, squeeze_73, mul_207, convolution_41, squeeze_76, add_133, mean_8, convolution_42, mul_216, convolution_43, mul_217, convolution_44, squeeze_79, add_139, convolution_45, squeeze_82, mul_232, convolution_46, squeeze_85, add_149, mean_9, convolution_47, mul_241, convolution_48, mul_242, convolution_49, squeeze_88, add_154, convolution_50, squeeze_91, mul_257, convolution_51, squeeze_94, add_164, mean_10, convolution_52, mul_266, convolution_53, mul_267, convolution_54, squeeze_97, add_170, convolution_55, squeeze_100, mul_282, convolution_56, squeeze_103, add_180, mean_11, convolution_57, mul_291, convolution_58, mul_292, convolution_59, squeeze_106, add_186, convolution_60, squeeze_109, mul_307, convolution_61, squeeze_112, add_196, mean_12, convolution_62, mul_316, convolution_63, mul_317, convolution_64, squeeze_115, add_202, convolution_65, squeeze_118, mul_332, convolution_66, squeeze_121, add_212, mean_13, convolution_67, mul_341, convolution_68, mul_342, convolution_69, squeeze_124, add_217, convolution_70, squeeze_127, mul_357, convolution_71, squeeze_130, add_227, mean_14, convolution_72, mul_366, convolution_73, mul_367, convolution_74, squeeze_133, add_233, convolution_75, squeeze_136, mul_382, convolution_76, squeeze_139, add_243, mean_15, convolution_77, mul_391, convolution_78, mul_392, convolution_79, squeeze_142, add_249, convolution_80, squeeze_145, mul_407, convolution_81, squeeze_148, add_259, mean_16, convolution_82, mul_416, convolution_83, mul_417, convolution_84, squeeze_151, add_265, convolution_85, squeeze_154, mul_432, convolution_86, squeeze_157, add_275, mean_17, convolution_87, mul_441, convolution_88, mul_442, convolution_89, squeeze_160, add_281, convolution_90, squeeze_163, mul_457, convolution_91, squeeze_166, add_291, mean_18, convolution_92, mul_466, convolution_93, mul_467, convolution_94, squeeze_169, add_296, convolution_95, squeeze_172, view, permute_1, mul_484, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_524, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_564, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_604, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_644, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_684, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_724, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_764, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_804, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_844, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_884, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_924, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_964, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_1004, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1044, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1084, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1124, unsqueeze_810, unsqueeze_822, unsqueeze_834, mul_1164, unsqueeze_846, unsqueeze_858, unsqueeze_870, mul_1204, unsqueeze_882, unsqueeze_894, unsqueeze_906, mul_1244, unsqueeze_918, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_13, (144, ), (1, ))
    assert_size_stride(primals_15, (144, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_19, (144, ), (1, ))
    assert_size_stride(primals_21, (144, ), (1, ))
    assert_size_stride(primals_23, (40, ), (1, ))
    assert_size_stride(primals_25, (240, ), (1, ))
    assert_size_stride(primals_27, (240, ), (1, ))
    assert_size_stride(primals_29, (40, ), (1, ))
    assert_size_stride(primals_31, (240, ), (1, ))
    assert_size_stride(primals_33, (240, ), (1, ))
    assert_size_stride(primals_35, (80, ), (1, ))
    assert_size_stride(primals_37, (480, ), (1, ))
    assert_size_stride(primals_39, (480, ), (1, ))
    assert_size_stride(primals_41, (80, ), (1, ))
    assert_size_stride(primals_43, (480, ), (1, ))
    assert_size_stride(primals_45, (480, ), (1, ))
    assert_size_stride(primals_47, (80, ), (1, ))
    assert_size_stride(primals_49, (480, ), (1, ))
    assert_size_stride(primals_51, (480, ), (1, ))
    assert_size_stride(primals_53, (80, ), (1, ))
    assert_size_stride(primals_55, (480, ), (1, ))
    assert_size_stride(primals_57, (480, ), (1, ))
    assert_size_stride(primals_59, (112, ), (1, ))
    assert_size_stride(primals_61, (672, ), (1, ))
    assert_size_stride(primals_63, (672, ), (1, ))
    assert_size_stride(primals_65, (112, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_69, (672, ), (1, ))
    assert_size_stride(primals_71, (112, ), (1, ))
    assert_size_stride(primals_73, (672, ), (1, ))
    assert_size_stride(primals_75, (672, ), (1, ))
    assert_size_stride(primals_77, (112, ), (1, ))
    assert_size_stride(primals_79, (672, ), (1, ))
    assert_size_stride(primals_81, (672, ), (1, ))
    assert_size_stride(primals_83, (192, ), (1, ))
    assert_size_stride(primals_85, (1152, ), (1, ))
    assert_size_stride(primals_87, (1152, ), (1, ))
    assert_size_stride(primals_89, (192, ), (1, ))
    assert_size_stride(primals_91, (1152, ), (1, ))
    assert_size_stride(primals_93, (1152, ), (1, ))
    assert_size_stride(primals_95, (192, ), (1, ))
    assert_size_stride(primals_97, (1152, ), (1, ))
    assert_size_stride(primals_99, (1152, ), (1, ))
    assert_size_stride(primals_101, (192, ), (1, ))
    assert_size_stride(primals_103, (1152, ), (1, ))
    assert_size_stride(primals_105, (1152, ), (1, ))
    assert_size_stride(primals_107, (192, ), (1, ))
    assert_size_stride(primals_109, (1152, ), (1, ))
    assert_size_stride(primals_111, (1152, ), (1, ))
    assert_size_stride(primals_113, (320, ), (1, ))
    assert_size_stride(primals_115, (1280, ), (1, ))
    assert_size_stride(primals_117, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_118, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_121, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_123, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_124, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_125, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_126, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_128, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_130, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_131, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_132, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_135, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_137, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_138, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_139, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_140, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_142, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_144, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_145, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_146, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_147, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_149, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_151, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_152, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_153, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_154, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_156, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_158, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_159, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_160, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_163, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_165, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_166, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_167, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_168, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_170, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_172, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_173, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_174, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_175, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_177, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_179, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_180, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_181, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_182, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_184, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_186, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_187, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_188, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_189, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_191, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_193, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_194, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_195, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_196, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_198, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_200, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_201, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_202, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_203, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_205, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_207, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_208, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_209, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_210, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_212, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_214, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_215, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_216, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_217, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_219, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_221, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_222, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_223, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_224, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_226, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_228, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_229, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_230, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_231, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_233, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_235, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_236, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_237, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_238, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_240, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_242, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_243, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_244, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_245, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_247, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_249, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_250, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_427, (8, 3, 192, 192), (110592, 1, 576, 3))
    assert_size_stride(convolution, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(mul_7, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(convolution_1, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(add_9, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(mean, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(convolution_2, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(mul_16, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_3, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(mul_17, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(convolution_4, (8, 16, 96, 96), (147456, 1, 1536, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_14, (8, 16, 96, 96), (147456, 1, 1536, 16))
    assert_size_stride(convolution_5, (8, 96, 96, 96), (884736, 1, 9216, 96))
    assert_size_stride(squeeze_10, (96, ), (1, ))
    assert_size_stride(mul_32, (8, 96, 96, 96), (884736, 1, 9216, 96))
    assert_size_stride(convolution_6, (8, 96, 48, 48), (221184, 1, 4608, 96))
    assert_size_stride(squeeze_13, (96, ), (1, ))
    assert_size_stride(add_24, (8, 96, 48, 48), (221184, 1, 4608, 96))
    assert_size_stride(mean_1, (8, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(convolution_7, (8, 4, 1, 1), (4, 1, 4, 4))
    assert_size_stride(mul_41, (8, 4, 1, 1), (4, 1, 4, 4))
    assert_size_stride(convolution_8, (8, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(mul_42, (8, 96, 48, 48), (221184, 1, 4608, 96))
    assert_size_stride(convolution_9, (8, 24, 48, 48), (55296, 1, 1152, 24))
    assert_size_stride(squeeze_16, (24, ), (1, ))
    assert_size_stride(add_29, (8, 24, 48, 48), (55296, 1, 1152, 24))
    assert_size_stride(convolution_10, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(squeeze_19, (144, ), (1, ))
    assert_size_stride(mul_57, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(convolution_11, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(squeeze_22, (144, ), (1, ))
    assert_size_stride(add_39, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(mean_2, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(convolution_12, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(mul_66, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_13, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(mul_67, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(convolution_14, (8, 24, 48, 48), (55296, 1, 1152, 24))
    assert_size_stride(squeeze_25, (24, ), (1, ))
    assert_size_stride(add_45, (8, 24, 48, 48), (55296, 1, 1152, 24))
    assert_size_stride(convolution_15, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(squeeze_28, (144, ), (1, ))
    assert_size_stride(mul_82, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(convolution_16, (8, 144, 24, 24), (82944, 1, 3456, 144))
    assert_size_stride(squeeze_31, (144, ), (1, ))
    assert_size_stride(add_55, (8, 144, 24, 24), (82944, 1, 3456, 144))
    assert_size_stride(mean_3, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(convolution_17, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(mul_91, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_18, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(mul_92, (8, 144, 24, 24), (82944, 1, 3456, 144))
    assert_size_stride(convolution_19, (8, 40, 24, 24), (23040, 1, 960, 40))
    assert_size_stride(squeeze_34, (40, ), (1, ))
    assert_size_stride(add_60, (8, 40, 24, 24), (23040, 1, 960, 40))
    assert_size_stride(convolution_20, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(squeeze_37, (240, ), (1, ))
    assert_size_stride(mul_107, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(convolution_21, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(squeeze_40, (240, ), (1, ))
    assert_size_stride(add_70, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(mean_4, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_22, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(mul_116, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(convolution_23, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_117, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(convolution_24, (8, 40, 24, 24), (23040, 1, 960, 40))
    assert_size_stride(squeeze_43, (40, ), (1, ))
    assert_size_stride(add_76, (8, 40, 24, 24), (23040, 1, 960, 40))
    assert_size_stride(convolution_25, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(squeeze_46, (240, ), (1, ))
    assert_size_stride(mul_132, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(convolution_26, (8, 240, 12, 12), (34560, 1, 2880, 240))
    assert_size_stride(squeeze_49, (240, ), (1, ))
    assert_size_stride(add_86, (8, 240, 12, 12), (34560, 1, 2880, 240))
    assert_size_stride(mean_5, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_27, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(mul_141, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(convolution_28, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_142, (8, 240, 12, 12), (34560, 1, 2880, 240))
    assert_size_stride(convolution_29, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(squeeze_52, (80, ), (1, ))
    assert_size_stride(add_91, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(convolution_30, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_55, (480, ), (1, ))
    assert_size_stride(mul_157, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_31, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_58, (480, ), (1, ))
    assert_size_stride(add_101, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(mean_6, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_32, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_166, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_33, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_167, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_34, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(squeeze_61, (80, ), (1, ))
    assert_size_stride(add_107, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(convolution_35, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_64, (480, ), (1, ))
    assert_size_stride(mul_182, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_36, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_67, (480, ), (1, ))
    assert_size_stride(add_117, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(mean_7, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_37, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_191, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_38, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_192, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_39, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(squeeze_70, (80, ), (1, ))
    assert_size_stride(add_123, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(convolution_40, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_73, (480, ), (1, ))
    assert_size_stride(mul_207, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_41, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_76, (480, ), (1, ))
    assert_size_stride(add_133, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(mean_8, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_42, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_216, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_43, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_217, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_44, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(squeeze_79, (80, ), (1, ))
    assert_size_stride(add_139, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(convolution_45, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_82, (480, ), (1, ))
    assert_size_stride(mul_232, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_46, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_85, (480, ), (1, ))
    assert_size_stride(add_149, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(mean_9, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_47, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_241, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_48, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_242, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_49, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(squeeze_88, (112, ), (1, ))
    assert_size_stride(add_154, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(convolution_50, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_91, (672, ), (1, ))
    assert_size_stride(mul_257, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_51, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_94, (672, ), (1, ))
    assert_size_stride(add_164, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(mean_10, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_52, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_266, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_53, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_267, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_54, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(squeeze_97, (112, ), (1, ))
    assert_size_stride(add_170, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(convolution_55, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_100, (672, ), (1, ))
    assert_size_stride(mul_282, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_56, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_103, (672, ), (1, ))
    assert_size_stride(add_180, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(mean_11, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_57, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_291, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_58, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_292, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_59, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(squeeze_106, (112, ), (1, ))
    assert_size_stride(add_186, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(convolution_60, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_109, (672, ), (1, ))
    assert_size_stride(mul_307, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_61, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_112, (672, ), (1, ))
    assert_size_stride(add_196, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(mean_12, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_62, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_316, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_63, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_317, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_64, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(squeeze_115, (112, ), (1, ))
    assert_size_stride(add_202, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(convolution_65, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_118, (672, ), (1, ))
    assert_size_stride(mul_332, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_66, (8, 672, 6, 6), (24192, 1, 4032, 672))
    assert_size_stride(squeeze_121, (672, ), (1, ))
    assert_size_stride(add_212, (8, 672, 6, 6), (24192, 1, 4032, 672))
    assert_size_stride(mean_13, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_67, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_341, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_68, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_342, (8, 672, 6, 6), (24192, 1, 4032, 672))
    assert_size_stride(convolution_69, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(squeeze_124, (192, ), (1, ))
    assert_size_stride(add_217, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(convolution_70, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_127, (1152, ), (1, ))
    assert_size_stride(mul_357, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_71, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_130, (1152, ), (1, ))
    assert_size_stride(add_227, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(mean_14, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_72, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_366, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_73, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_367, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_74, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(squeeze_133, (192, ), (1, ))
    assert_size_stride(add_233, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(convolution_75, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_136, (1152, ), (1, ))
    assert_size_stride(mul_382, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_76, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_139, (1152, ), (1, ))
    assert_size_stride(add_243, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(mean_15, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_77, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_391, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_78, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_392, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_79, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(squeeze_142, (192, ), (1, ))
    assert_size_stride(add_249, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(convolution_80, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_145, (1152, ), (1, ))
    assert_size_stride(mul_407, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_81, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_148, (1152, ), (1, ))
    assert_size_stride(add_259, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(mean_16, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_82, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_416, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_83, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_417, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_84, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(squeeze_151, (192, ), (1, ))
    assert_size_stride(add_265, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(convolution_85, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_154, (1152, ), (1, ))
    assert_size_stride(mul_432, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_86, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_157, (1152, ), (1, ))
    assert_size_stride(add_275, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(mean_17, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_87, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_441, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_88, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_442, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_89, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(squeeze_160, (192, ), (1, ))
    assert_size_stride(add_281, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(convolution_90, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_163, (1152, ), (1, ))
    assert_size_stride(mul_457, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_91, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_166, (1152, ), (1, ))
    assert_size_stride(add_291, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(mean_18, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_92, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_466, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_93, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_467, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_94, (8, 320, 6, 6), (11520, 1, 1920, 320))
    assert_size_stride(squeeze_169, (320, ), (1, ))
    assert_size_stride(add_296, (8, 320, 6, 6), (11520, 1, 1920, 320))
    assert_size_stride(convolution_95, (8, 1280, 6, 6), (46080, 1, 7680, 1280))
    assert_size_stride(squeeze_172, (1280, ), (1, ))
    assert_size_stride(view, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(mul_484, (8, 1280, 6, 6), (46080, 1, 7680, 1280))
    assert_size_stride(unsqueeze_234, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_246, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_524, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(unsqueeze_270, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_564, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(unsqueeze_306, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_318, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_604, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(unsqueeze_342, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_644, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(unsqueeze_378, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_684, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(unsqueeze_414, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_724, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(unsqueeze_450, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_764, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(unsqueeze_486, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_804, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(unsqueeze_522, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_844, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(unsqueeze_558, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_884, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(unsqueeze_594, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_606, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_924, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(unsqueeze_630, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_964, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(unsqueeze_666, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_678, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_1004, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(unsqueeze_702, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(mul_1044, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(unsqueeze_738, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_750, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(mul_1084, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(unsqueeze_774, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_786, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_798, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(mul_1124, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(unsqueeze_810, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_822, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_834, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(mul_1164, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(unsqueeze_846, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_858, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_870, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(mul_1204, (8, 96, 96, 96), (884736, 1, 9216, 96))
    assert_size_stride(unsqueeze_882, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_894, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_906, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_1244, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(unsqueeze_918, (1, 32, 1, 1), (32, 1, 1, 1))
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
        buf3 = empty_strided((1280, 3), (1, 1280), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1280, 3), (1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_div_mul_native_batch_norm_backward_1.run(buf0, mul_484, convolution_95, unsqueeze_234, buf3, buf5, 3840, 96, grid=grid(3840), stream=stream0)
        buf4 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_mul_native_batch_norm_backward_2.run(buf3, buf4, 1280, 3, grid=grid(1280), stream=stream0)
        del buf3
        buf6 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf7 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_mul_native_batch_norm_backward_3.run(buf5, squeeze_172, buf6, buf7, 1280, 3, grid=grid(1280), stream=stream0)
        buf8 = empty((8, 1280, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_4.run(buf0, mul_484, convolution_95, unsqueeze_234, buf6, squeeze_172, buf4, primals_115, buf8, 288, 1280, grid=grid(288, 1280), stream=stream0)
        del buf0
        del buf6
        del convolution_95
        del mul_484
        del primals_115
        del squeeze_172
        del unsqueeze_234
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        buf9 = aten.convolution_backward(buf8, add_296, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_296
        del buf8
        del primals_250
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_5.run(buf10, buf12, 320, 288, grid=grid(320), stream=stream0)
        buf13 = empty_strided((320, 3), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_6.run(buf10, convolution_94, unsqueeze_246, buf13, 960, 96, grid=grid(960), stream=stream0)
        buf14 = empty((320, ), device='cuda', dtype=torch.float32)
        buf15 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_7.run(buf13, squeeze_169, buf14, buf15, 320, 3, grid=grid(320), stream=stream0)
        del buf13
        buf16 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_8.run(buf16, convolution_94, unsqueeze_246, buf14, squeeze_169, buf12, primals_113, 2560, 36, grid=grid(2560, 36), stream=stream0)
        del buf14
        del convolution_94
        del primals_113
        del squeeze_169
        del unsqueeze_246
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf17 = aten.convolution_backward(buf16, mul_467, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_467
        del primals_249
        buf18 = buf17[0]
        buf19 = buf17[1]
        del buf17
        buf20 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cuda', dtype=torch.float32)
        buf21 = reinterpret_tensor(buf20, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf20  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_309], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf21, buf18, add_291, convolution_93, 9216, 36, grid=grid(9216), stream=stream0)
        buf22 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf21, buf22, 1152, 8, grid=grid(1152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf23 = aten.convolution_backward(buf21, mul_466, primals_247, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf21
        del mul_466
        del primals_247
        buf24 = buf23[0]
        buf25 = buf23[1]
        del buf23
        buf26 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf26, convolution_92, 384, grid=grid(384), stream=stream0)
        del convolution_92
        buf27 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf26, buf27, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf28 = aten.convolution_backward(buf26, mean_18, primals_245, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf26
        del mean_18
        del primals_245
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = empty_strided((1152, 3), (1, 1152), device='cuda', dtype=torch.float32)
        buf33 = empty_strided((1152, 3), (1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13.run(buf18, convolution_93, buf29, add_291, convolution_91, unsqueeze_258, buf31, buf33, 3456, 96, grid=grid(3456), stream=stream0)
        buf32 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf31, buf32, 1152, 3, grid=grid(1152), stream=stream0)
        buf34 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf36 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf33, squeeze_166, buf34, buf36, 1152, 3, grid=grid(1152), stream=stream0)
        buf35 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16.run(buf18, convolution_93, buf29, add_291, convolution_91, unsqueeze_258, buf34, squeeze_166, buf32, buf35, 288, 1152, grid=grid(288, 1152), stream=stream0)
        del add_291
        del convolution_91
        del convolution_93
        del unsqueeze_258
        buf37 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_17.run(buf35, squeeze_166, primals_111, buf37, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del buf35
        del primals_111
        del squeeze_166
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf38 = aten.convolution_backward(buf37, mul_457, primals_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False])
        del buf37
        del mul_457
        del primals_244
        buf39 = buf38[0]
        buf40 = buf38[1]
        del buf38
        buf41 = buf33; del buf33  # reuse
        buf43 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_18.run(buf39, mul_524, convolution_90, unsqueeze_270, buf41, buf43, 3456, 96, grid=grid(3456), stream=stream0)
        buf42 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf41, buf42, 1152, 3, grid=grid(1152), stream=stream0)
        buf44 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf45 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf43, squeeze_163, buf44, buf45, 1152, 3, grid=grid(1152), stream=stream0)
        buf46 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19.run(buf46, mul_524, convolution_90, unsqueeze_270, buf44, squeeze_163, buf42, primals_109, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del convolution_90
        del mul_524
        del primals_109
        del squeeze_163
        del unsqueeze_270
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf47 = aten.convolution_backward(buf46, add_281, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_281
        del primals_243
        buf48 = buf47[0]
        buf49 = buf47[1]
        del buf47
        buf50 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_20.run(buf48, buf50, 192, 288, grid=grid(192), stream=stream0)
        buf51 = empty_strided((192, 3), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_21.run(buf48, convolution_89, unsqueeze_282, buf51, 576, 96, grid=grid(576), stream=stream0)
        buf52 = empty((192, ), device='cuda', dtype=torch.float32)
        buf53 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_22.run(buf51, squeeze_160, buf52, buf53, 192, 3, grid=grid(192), stream=stream0)
        buf54 = empty((8, 192, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_23.run(buf48, convolution_89, unsqueeze_282, buf52, squeeze_160, buf50, primals_107, buf54, 1536, 36, grid=grid(1536, 36), stream=stream0)
        del convolution_89
        del primals_107
        del squeeze_160
        del unsqueeze_282
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf55 = aten.convolution_backward(buf54, mul_442, primals_242, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_442
        del primals_242
        buf56 = buf55[0]
        buf57 = buf55[1]
        del buf55
        buf58 = reinterpret_tensor(buf29, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf29  # reuse
        buf59 = reinterpret_tensor(buf58, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf58  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate, x_292], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf59, buf56, add_275, convolution_88, 9216, 36, grid=grid(9216), stream=stream0)
        buf60 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf59, buf60, 1152, 8, grid=grid(1152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf61 = aten.convolution_backward(buf59, mul_441, primals_240, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf59
        del mul_441
        del primals_240
        buf62 = buf61[0]
        buf63 = buf61[1]
        del buf61
        buf64 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf64, convolution_87, 384, grid=grid(384), stream=stream0)
        del convolution_87
        buf65 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf64, buf65, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf66 = aten.convolution_backward(buf64, mean_17, primals_238, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf64
        del mean_17
        del primals_238
        buf67 = buf66[0]
        buf68 = buf66[1]
        del buf66
        buf69 = buf43; del buf43  # reuse
        buf71 = buf41; del buf41  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13.run(buf56, convolution_88, buf67, add_275, convolution_86, unsqueeze_294, buf69, buf71, 3456, 96, grid=grid(3456), stream=stream0)
        buf70 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf69, buf70, 1152, 3, grid=grid(1152), stream=stream0)
        buf72 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf74 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf71, squeeze_157, buf72, buf74, 1152, 3, grid=grid(1152), stream=stream0)
        buf73 = reinterpret_tensor(buf46, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf46  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16.run(buf56, convolution_88, buf67, add_275, convolution_86, unsqueeze_294, buf72, squeeze_157, buf70, buf73, 288, 1152, grid=grid(288, 1152), stream=stream0)
        del add_275
        del convolution_86
        del convolution_88
        del unsqueeze_294
        buf75 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_17.run(buf73, squeeze_157, primals_105, buf75, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del buf73
        del primals_105
        del squeeze_157
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf76 = aten.convolution_backward(buf75, mul_432, primals_237, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
        del buf75
        del mul_432
        del primals_237
        buf77 = buf76[0]
        buf78 = buf76[1]
        del buf76
        buf79 = buf71; del buf71  # reuse
        buf81 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_18.run(buf77, mul_564, convolution_85, unsqueeze_306, buf79, buf81, 3456, 96, grid=grid(3456), stream=stream0)
        buf80 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf79, buf80, 1152, 3, grid=grid(1152), stream=stream0)
        buf82 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf83 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf81, squeeze_154, buf82, buf83, 1152, 3, grid=grid(1152), stream=stream0)
        buf84 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19.run(buf84, mul_564, convolution_85, unsqueeze_306, buf82, squeeze_154, buf80, primals_103, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del convolution_85
        del mul_564
        del primals_103
        del squeeze_154
        del unsqueeze_306
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf85 = aten.convolution_backward(buf84, add_265, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_265
        del primals_236
        buf86 = buf85[0]
        buf87 = buf85[1]
        del buf85
        buf88 = buf52; del buf52  # reuse
        buf89 = empty((192, ), device='cuda', dtype=torch.float32)
        buf90 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_24.run(buf48, buf86, convolution_84, unsqueeze_318, squeeze_151, buf88, buf89, buf90, 192, 288, grid=grid(192), stream=stream0)
        buf91 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_25.run(buf48, buf86, convolution_84, unsqueeze_318, buf89, squeeze_151, buf88, primals_101, buf91, 1536, 36, grid=grid(1536, 36), stream=stream0)
        del convolution_84
        del primals_101
        del squeeze_151
        del unsqueeze_318
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf92 = aten.convolution_backward(buf91, mul_417, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_417
        del primals_235
        buf93 = buf92[0]
        buf94 = buf92[1]
        del buf92
        buf95 = reinterpret_tensor(buf67, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf67  # reuse
        buf96 = reinterpret_tensor(buf95, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf95  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_275], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf96, buf93, add_259, convolution_83, 9216, 36, grid=grid(9216), stream=stream0)
        buf97 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf96, buf97, 1152, 8, grid=grid(1152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf98 = aten.convolution_backward(buf96, mul_416, primals_233, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf96
        del mul_416
        del primals_233
        buf99 = buf98[0]
        buf100 = buf98[1]
        del buf98
        buf101 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf101, convolution_82, 384, grid=grid(384), stream=stream0)
        del convolution_82
        buf102 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf101, buf102, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf103 = aten.convolution_backward(buf101, mean_16, primals_231, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf101
        del mean_16
        del primals_231
        buf104 = buf103[0]
        buf105 = buf103[1]
        del buf103
        buf106 = buf81; del buf81  # reuse
        buf108 = buf79; del buf79  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13.run(buf93, convolution_83, buf104, add_259, convolution_81, unsqueeze_330, buf106, buf108, 3456, 96, grid=grid(3456), stream=stream0)
        buf107 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf106, buf107, 1152, 3, grid=grid(1152), stream=stream0)
        buf109 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf111 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf108, squeeze_148, buf109, buf111, 1152, 3, grid=grid(1152), stream=stream0)
        buf110 = reinterpret_tensor(buf84, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf84  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16.run(buf93, convolution_83, buf104, add_259, convolution_81, unsqueeze_330, buf109, squeeze_148, buf107, buf110, 288, 1152, grid=grid(288, 1152), stream=stream0)
        del add_259
        del convolution_81
        del convolution_83
        del unsqueeze_330
        buf112 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_17.run(buf110, squeeze_148, primals_99, buf112, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del buf110
        del primals_99
        del squeeze_148
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf113 = aten.convolution_backward(buf112, mul_407, primals_230, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
        del buf112
        del mul_407
        del primals_230
        buf114 = buf113[0]
        buf115 = buf113[1]
        del buf113
        buf116 = buf108; del buf108  # reuse
        buf118 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_18.run(buf114, mul_604, convolution_80, unsqueeze_342, buf116, buf118, 3456, 96, grid=grid(3456), stream=stream0)
        buf117 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf116, buf117, 1152, 3, grid=grid(1152), stream=stream0)
        buf119 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf120 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf118, squeeze_145, buf119, buf120, 1152, 3, grid=grid(1152), stream=stream0)
        buf121 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19.run(buf121, mul_604, convolution_80, unsqueeze_342, buf119, squeeze_145, buf117, primals_97, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del convolution_80
        del mul_604
        del primals_97
        del squeeze_145
        del unsqueeze_342
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf122 = aten.convolution_backward(buf121, add_249, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_249
        del primals_229
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        buf125 = buf89; del buf89  # reuse
        buf126 = empty((192, ), device='cuda', dtype=torch.float32)
        buf128 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_26.run(buf48, buf86, buf123, convolution_79, unsqueeze_354, squeeze_142, buf125, buf126, buf128, 192, 288, grid=grid(192), stream=stream0)
        buf127 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_27.run(buf48, buf86, buf123, convolution_79, unsqueeze_354, buf126, squeeze_142, buf125, primals_95, buf127, 1536, 36, grid=grid(1536, 36), stream=stream0)
        del convolution_79
        del primals_95
        del squeeze_142
        del unsqueeze_354
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf129 = aten.convolution_backward(buf127, mul_392, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_392
        del primals_228
        buf130 = buf129[0]
        buf131 = buf129[1]
        del buf129
        buf132 = reinterpret_tensor(buf104, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf104  # reuse
        buf133 = reinterpret_tensor(buf132, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf132  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_258], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf133, buf130, add_243, convolution_78, 9216, 36, grid=grid(9216), stream=stream0)
        buf134 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf133, buf134, 1152, 8, grid=grid(1152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf135 = aten.convolution_backward(buf133, mul_391, primals_226, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf133
        del mul_391
        del primals_226
        buf136 = buf135[0]
        buf137 = buf135[1]
        del buf135
        buf138 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf138, convolution_77, 384, grid=grid(384), stream=stream0)
        del convolution_77
        buf139 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf138, buf139, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf140 = aten.convolution_backward(buf138, mean_15, primals_224, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf138
        del mean_15
        del primals_224
        buf141 = buf140[0]
        buf142 = buf140[1]
        del buf140
        buf143 = buf118; del buf118  # reuse
        buf145 = buf116; del buf116  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13.run(buf130, convolution_78, buf141, add_243, convolution_76, unsqueeze_366, buf143, buf145, 3456, 96, grid=grid(3456), stream=stream0)
        buf144 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf143, buf144, 1152, 3, grid=grid(1152), stream=stream0)
        buf146 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf148 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf145, squeeze_139, buf146, buf148, 1152, 3, grid=grid(1152), stream=stream0)
        buf147 = reinterpret_tensor(buf121, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf121  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16.run(buf130, convolution_78, buf141, add_243, convolution_76, unsqueeze_366, buf146, squeeze_139, buf144, buf147, 288, 1152, grid=grid(288, 1152), stream=stream0)
        del add_243
        del convolution_76
        del convolution_78
        del unsqueeze_366
        buf149 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_17.run(buf147, squeeze_139, primals_93, buf149, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del buf147
        del primals_93
        del squeeze_139
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf150 = aten.convolution_backward(buf149, mul_382, primals_223, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
        del buf149
        del mul_382
        del primals_223
        buf151 = buf150[0]
        buf152 = buf150[1]
        del buf150
        buf153 = buf145; del buf145  # reuse
        buf155 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_18.run(buf151, mul_644, convolution_75, unsqueeze_378, buf153, buf155, 3456, 96, grid=grid(3456), stream=stream0)
        buf154 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf153, buf154, 1152, 3, grid=grid(1152), stream=stream0)
        buf156 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf157 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf155, squeeze_136, buf156, buf157, 1152, 3, grid=grid(1152), stream=stream0)
        buf158 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19.run(buf158, mul_644, convolution_75, unsqueeze_378, buf156, squeeze_136, buf154, primals_91, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del convolution_75
        del mul_644
        del primals_91
        del squeeze_136
        del unsqueeze_378
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf159 = aten.convolution_backward(buf158, add_233, primals_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_233
        del primals_222
        buf160 = buf159[0]
        buf161 = buf159[1]
        del buf159
        buf162 = buf126; del buf126  # reuse
        buf163 = empty((192, ), device='cuda', dtype=torch.float32)
        buf165 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_28.run(buf48, buf86, buf123, buf160, convolution_74, unsqueeze_390, squeeze_133, buf162, buf163, buf165, 192, 288, grid=grid(192), stream=stream0)
        buf164 = buf127; del buf127  # reuse
        buf166 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_29.run(buf166, buf48, buf86, buf123, buf160, convolution_74, unsqueeze_390, buf163, squeeze_133, buf162, primals_89, 1536, 36, grid=grid(1536, 36), stream=stream0)
        del convolution_74
        del primals_89
        del squeeze_133
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf167 = aten.convolution_backward(buf166, mul_367, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf166
        del mul_367
        del primals_221
        buf168 = buf167[0]
        buf169 = buf167[1]
        del buf167
        buf170 = reinterpret_tensor(buf141, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf141  # reuse
        buf171 = reinterpret_tensor(buf170, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf170  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_241], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf171, buf168, add_227, convolution_73, 9216, 36, grid=grid(9216), stream=stream0)
        buf172 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf171, buf172, 1152, 8, grid=grid(1152), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf173 = aten.convolution_backward(buf171, mul_366, primals_219, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf171
        del mul_366
        del primals_219
        buf174 = buf173[0]
        buf175 = buf173[1]
        del buf173
        buf176 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf176, convolution_72, 384, grid=grid(384), stream=stream0)
        del convolution_72
        buf177 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf176, buf177, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf178 = aten.convolution_backward(buf176, mean_14, primals_217, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf176
        del mean_14
        del primals_217
        buf179 = buf178[0]
        buf180 = buf178[1]
        del buf178
        buf181 = buf155; del buf155  # reuse
        buf183 = buf153; del buf153  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_13.run(buf168, convolution_73, buf179, add_227, convolution_71, unsqueeze_402, buf181, buf183, 3456, 96, grid=grid(3456), stream=stream0)
        buf182 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf181, buf182, 1152, 3, grid=grid(1152), stream=stream0)
        buf184 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf186 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf183, squeeze_130, buf184, buf186, 1152, 3, grid=grid(1152), stream=stream0)
        buf185 = reinterpret_tensor(buf158, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf158  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_16.run(buf168, convolution_73, buf179, add_227, convolution_71, unsqueeze_402, buf184, squeeze_130, buf182, buf185, 288, 1152, grid=grid(288, 1152), stream=stream0)
        del add_227
        del convolution_71
        del convolution_73
        del unsqueeze_402
        buf187 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_17.run(buf185, squeeze_130, primals_87, buf187, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del buf185
        del primals_87
        del squeeze_130
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf188 = aten.convolution_backward(buf187, mul_357, primals_216, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
        del buf187
        del mul_357
        del primals_216
        buf189 = buf188[0]
        buf190 = buf188[1]
        del buf188
        buf191 = buf183; del buf183  # reuse
        buf193 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_18.run(buf189, mul_684, convolution_70, unsqueeze_414, buf191, buf193, 3456, 96, grid=grid(3456), stream=stream0)
        buf192 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14.run(buf191, buf192, 1152, 3, grid=grid(1152), stream=stream0)
        del buf191
        buf194 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf195 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_15.run(buf193, squeeze_127, buf194, buf195, 1152, 3, grid=grid(1152), stream=stream0)
        buf196 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_19.run(buf196, mul_684, convolution_70, unsqueeze_414, buf194, squeeze_127, buf192, primals_85, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del convolution_70
        del mul_684
        del primals_85
        del squeeze_127
        del unsqueeze_414
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf197 = aten.convolution_backward(buf196, add_217, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_217
        del buf196
        del primals_215
        buf198 = buf197[0]
        buf199 = buf197[1]
        del buf197
        buf200 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_30.run(buf200, buf48, buf86, buf160, buf198, 55296, grid=grid(55296), stream=stream0)
        del buf160
        del buf198
        del buf48
        del buf86
        buf201 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_20.run(buf200, buf201, 192, 288, grid=grid(192), stream=stream0)
        buf202 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_21.run(buf200, convolution_69, unsqueeze_426, buf202, 576, 96, grid=grid(576), stream=stream0)
        buf203 = empty((192, ), device='cuda', dtype=torch.float32)
        buf204 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_22.run(buf202, squeeze_124, buf203, buf204, 192, 3, grid=grid(192), stream=stream0)
        del buf202
        buf205 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_31.run(buf205, convolution_69, unsqueeze_426, buf203, squeeze_124, buf201, primals_83, 1536, 36, grid=grid(1536, 36), stream=stream0)
        del buf203
        del convolution_69
        del primals_83
        del squeeze_124
        del unsqueeze_426
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf206 = aten.convolution_backward(buf205, mul_342, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_342
        del primals_214
        buf207 = buf206[0]
        buf208 = buf206[1]
        del buf206
        buf209 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf210 = reinterpret_tensor(buf209, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf209  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_225], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_32.run(buf210, buf207, add_212, convolution_68, 5376, 36, grid=grid(5376), stream=stream0)
        buf211 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_33.run(buf210, buf211, 672, 8, grid=grid(672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf212 = aten.convolution_backward(buf210, mul_341, primals_212, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf210
        del mul_341
        del primals_212
        buf213 = buf212[0]
        buf214 = buf212[1]
        del buf212
        buf215 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_34.run(buf215, convolution_67, 224, grid=grid(224), stream=stream0)
        del convolution_67
        buf216 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_35.run(buf215, buf216, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf217 = aten.convolution_backward(buf215, mean_13, primals_210, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf215
        del mean_13
        del primals_210
        buf218 = buf217[0]
        buf219 = buf217[1]
        del buf217
        buf220 = empty_strided((672, 3), (1, 672), device='cuda', dtype=torch.float32)
        buf222 = empty_strided((672, 3), (1, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_36.run(buf207, convolution_68, buf218, add_212, convolution_66, unsqueeze_438, buf220, buf222, 2016, 96, grid=grid(2016), stream=stream0)
        buf221 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_37.run(buf220, buf221, 672, 3, grid=grid(672), stream=stream0)
        del buf220
        buf223 = empty((672, ), device='cuda', dtype=torch.float32)
        buf225 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_38.run(buf222, squeeze_121, buf223, buf225, 672, 3, grid=grid(672), stream=stream0)
        del buf222
        buf224 = empty_strided((8, 672, 6, 6), (24192, 1, 4032, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_39.run(buf207, convolution_68, buf218, add_212, convolution_66, unsqueeze_438, buf223, squeeze_121, buf221, buf224, 288, 672, grid=grid(288, 672), stream=stream0)
        del add_212
        del convolution_66
        del convolution_68
        del unsqueeze_438
        buf226 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_40.run(buf224, squeeze_121, primals_81, buf226, 5376, 36, grid=grid(5376, 36), stream=stream0)
        del buf224
        del primals_81
        del squeeze_121
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf227 = aten.convolution_backward(buf226, mul_332, primals_209, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
        del buf226
        del mul_332
        del primals_209
        buf228 = buf227[0]
        buf229 = buf227[1]
        del buf227
        buf230 = empty((672, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_41.run(buf228, mul_724, buf230, 6048, 128, grid=grid(6048), stream=stream0)
        buf231 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_42.run(buf230, buf231, 672, 9, grid=grid(672), stream=stream0)
        buf232 = reinterpret_tensor(buf230, (672, 9), (1, 672), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_43.run(buf228, mul_724, convolution_65, unsqueeze_450, buf232, 6048, 128, grid=grid(6048), stream=stream0)
        buf233 = empty((672, ), device='cuda', dtype=torch.float32)
        buf234 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_44.run(buf232, squeeze_118, buf233, buf234, 672, 9, grid=grid(672), stream=stream0)
        buf235 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_45.run(buf235, mul_724, convolution_65, unsqueeze_450, buf233, squeeze_118, buf231, primals_79, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del convolution_65
        del mul_724
        del primals_79
        del squeeze_118
        del unsqueeze_450
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf236 = aten.convolution_backward(buf235, add_202, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_202
        del primals_208
        buf237 = buf236[0]
        buf238 = buf236[1]
        del buf236
        buf239 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_46.run(buf237, buf239, 112, 1152, grid=grid(112), stream=stream0)
        buf240 = empty((112, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_47.run(buf237, convolution_64, unsqueeze_462, buf240, 1008, 128, grid=grid(1008), stream=stream0)
        buf241 = empty((112, ), device='cuda', dtype=torch.float32)
        buf242 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_48.run(buf240, squeeze_115, buf241, buf242, 112, 9, grid=grid(112), stream=stream0)
        del buf240
        buf243 = empty((8, 112, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_49.run(buf237, convolution_64, unsqueeze_462, buf241, squeeze_115, buf239, primals_77, buf243, 896, 144, grid=grid(896, 144), stream=stream0)
        del convolution_64
        del primals_77
        del squeeze_115
        del unsqueeze_462
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf244 = aten.convolution_backward(buf243, mul_317, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_317
        del primals_207
        buf245 = buf244[0]
        buf246 = buf244[1]
        del buf244
        buf247 = empty_strided((8, 672, 1, 1, 2), (1344, 2, 10752, 10752, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_208], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_50.run(buf245, add_196, buf247, 10752, 72, grid=grid(10752), stream=stream0)
        buf248 = reinterpret_tensor(buf218, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf218  # reuse
        buf249 = reinterpret_tensor(buf248, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf248  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_208], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_51.run(buf249, buf247, convolution_63, 5376, 2, grid=grid(5376), stream=stream0)
        buf250 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_33.run(buf249, buf250, 672, 8, grid=grid(672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf251 = aten.convolution_backward(buf249, mul_316, primals_205, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf249
        del mul_316
        del primals_205
        buf252 = buf251[0]
        buf253 = buf251[1]
        del buf251
        buf254 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_34.run(buf254, convolution_62, 224, grid=grid(224), stream=stream0)
        del convolution_62
        buf255 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_35.run(buf254, buf255, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf256 = aten.convolution_backward(buf254, mean_12, primals_203, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf254
        del mean_12
        del primals_203
        buf257 = buf256[0]
        buf258 = buf256[1]
        del buf256
        buf259 = reinterpret_tensor(buf232, (672, 9), (9, 1), 0); del buf232  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52.run(buf245, convolution_63, buf257, add_196, buf259, 6048, 128, grid=grid(6048), stream=stream0)
        buf260 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_42.run(buf259, buf260, 672, 9, grid=grid(672), stream=stream0)
        buf261 = reinterpret_tensor(buf259, (672, 9), (1, 672), 0); del buf259  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_53.run(buf245, convolution_63, buf257, add_196, convolution_61, unsqueeze_474, buf261, 6048, 128, grid=grid(6048), stream=stream0)
        buf262 = empty((672, ), device='cuda', dtype=torch.float32)
        buf264 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_44.run(buf261, squeeze_112, buf262, buf264, 672, 9, grid=grid(672), stream=stream0)
        buf263 = reinterpret_tensor(buf235, (8, 672, 12, 12), (96768, 1, 8064, 672), 0); del buf235  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_54.run(buf245, convolution_63, buf257, add_196, convolution_61, unsqueeze_474, buf262, squeeze_112, buf260, buf263, 1152, 672, grid=grid(1152, 672), stream=stream0)
        del add_196
        del convolution_61
        del convolution_63
        del unsqueeze_474
        buf265 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf263, squeeze_112, primals_75, buf265, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del buf263
        del primals_75
        del squeeze_112
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf266 = aten.convolution_backward(buf265, mul_307, primals_202, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
        del buf265
        del mul_307
        del primals_202
        buf267 = buf266[0]
        buf268 = buf266[1]
        del buf266
        buf269 = reinterpret_tensor(buf261, (672, 9), (9, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_41.run(buf267, mul_764, buf269, 6048, 128, grid=grid(6048), stream=stream0)
        buf270 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_42.run(buf269, buf270, 672, 9, grid=grid(672), stream=stream0)
        buf271 = reinterpret_tensor(buf269, (672, 9), (1, 672), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_43.run(buf267, mul_764, convolution_60, unsqueeze_486, buf271, 6048, 128, grid=grid(6048), stream=stream0)
        buf272 = empty((672, ), device='cuda', dtype=torch.float32)
        buf273 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_44.run(buf271, squeeze_109, buf272, buf273, 672, 9, grid=grid(672), stream=stream0)
        buf274 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_45.run(buf274, mul_764, convolution_60, unsqueeze_486, buf272, squeeze_109, buf270, primals_73, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del convolution_60
        del mul_764
        del primals_73
        del squeeze_109
        del unsqueeze_486
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf275 = aten.convolution_backward(buf274, add_186, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_186
        del primals_201
        buf276 = buf275[0]
        buf277 = buf275[1]
        del buf275
        buf278 = buf241; del buf241  # reuse
        buf279 = empty((112, ), device='cuda', dtype=torch.float32)
        buf280 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_56.run(buf237, buf276, convolution_59, unsqueeze_498, squeeze_106, buf278, buf279, buf280, 112, 1152, grid=grid(112), stream=stream0)
        buf281 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_57.run(buf237, buf276, convolution_59, unsqueeze_498, buf279, squeeze_106, buf278, primals_71, buf281, 896, 144, grid=grid(896, 144), stream=stream0)
        del convolution_59
        del primals_71
        del squeeze_106
        del unsqueeze_498
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf282 = aten.convolution_backward(buf281, mul_292, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_292
        del primals_200
        buf283 = buf282[0]
        buf284 = buf282[1]
        del buf282
        buf285 = buf247; del buf247  # reuse
        # Source Nodes: [x_191], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_50.run(buf283, add_180, buf285, 10752, 72, grid=grid(10752), stream=stream0)
        buf286 = reinterpret_tensor(buf257, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf257  # reuse
        buf287 = reinterpret_tensor(buf286, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf286  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_191], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_51.run(buf287, buf285, convolution_58, 5376, 2, grid=grid(5376), stream=stream0)
        buf288 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_33.run(buf287, buf288, 672, 8, grid=grid(672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf289 = aten.convolution_backward(buf287, mul_291, primals_198, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf287
        del mul_291
        del primals_198
        buf290 = buf289[0]
        buf291 = buf289[1]
        del buf289
        buf292 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_34.run(buf292, convolution_57, 224, grid=grid(224), stream=stream0)
        del convolution_57
        buf293 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_35.run(buf292, buf293, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf294 = aten.convolution_backward(buf292, mean_11, primals_196, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf292
        del mean_11
        del primals_196
        buf295 = buf294[0]
        buf296 = buf294[1]
        del buf294
        buf297 = reinterpret_tensor(buf271, (672, 9), (9, 1), 0); del buf271  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52.run(buf283, convolution_58, buf295, add_180, buf297, 6048, 128, grid=grid(6048), stream=stream0)
        buf298 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_42.run(buf297, buf298, 672, 9, grid=grid(672), stream=stream0)
        buf299 = reinterpret_tensor(buf297, (672, 9), (1, 672), 0); del buf297  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_53.run(buf283, convolution_58, buf295, add_180, convolution_56, unsqueeze_510, buf299, 6048, 128, grid=grid(6048), stream=stream0)
        buf300 = empty((672, ), device='cuda', dtype=torch.float32)
        buf302 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_44.run(buf299, squeeze_103, buf300, buf302, 672, 9, grid=grid(672), stream=stream0)
        buf301 = reinterpret_tensor(buf274, (8, 672, 12, 12), (96768, 1, 8064, 672), 0); del buf274  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_54.run(buf283, convolution_58, buf295, add_180, convolution_56, unsqueeze_510, buf300, squeeze_103, buf298, buf301, 1152, 672, grid=grid(1152, 672), stream=stream0)
        del add_180
        del convolution_56
        del convolution_58
        del unsqueeze_510
        buf303 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf301, squeeze_103, primals_69, buf303, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del buf301
        del primals_69
        del squeeze_103
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf304 = aten.convolution_backward(buf303, mul_282, primals_195, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
        del buf303
        del mul_282
        del primals_195
        buf305 = buf304[0]
        buf306 = buf304[1]
        del buf304
        buf307 = reinterpret_tensor(buf299, (672, 9), (9, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_41.run(buf305, mul_804, buf307, 6048, 128, grid=grid(6048), stream=stream0)
        buf308 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_42.run(buf307, buf308, 672, 9, grid=grid(672), stream=stream0)
        buf309 = reinterpret_tensor(buf307, (672, 9), (1, 672), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_43.run(buf305, mul_804, convolution_55, unsqueeze_522, buf309, 6048, 128, grid=grid(6048), stream=stream0)
        buf310 = empty((672, ), device='cuda', dtype=torch.float32)
        buf311 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_44.run(buf309, squeeze_100, buf310, buf311, 672, 9, grid=grid(672), stream=stream0)
        buf312 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_45.run(buf312, mul_804, convolution_55, unsqueeze_522, buf310, squeeze_100, buf308, primals_67, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del convolution_55
        del mul_804
        del primals_67
        del squeeze_100
        del unsqueeze_522
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf313 = aten.convolution_backward(buf312, add_170, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_170
        del primals_194
        buf314 = buf313[0]
        buf315 = buf313[1]
        del buf313
        buf316 = buf279; del buf279  # reuse
        buf317 = empty((112, ), device='cuda', dtype=torch.float32)
        buf319 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_58.run(buf237, buf276, buf314, convolution_54, unsqueeze_534, squeeze_97, buf316, buf317, buf319, 112, 1152, grid=grid(112), stream=stream0)
        buf318 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_59.run(buf237, buf276, buf314, convolution_54, unsqueeze_534, buf317, squeeze_97, buf316, primals_65, buf318, 896, 144, grid=grid(896, 144), stream=stream0)
        del convolution_54
        del primals_65
        del squeeze_97
        del unsqueeze_534
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf320 = aten.convolution_backward(buf318, mul_267, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf318
        del mul_267
        del primals_193
        buf321 = buf320[0]
        buf322 = buf320[1]
        del buf320
        buf323 = buf285; del buf285  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_50.run(buf321, add_164, buf323, 10752, 72, grid=grid(10752), stream=stream0)
        buf324 = reinterpret_tensor(buf295, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf295  # reuse
        buf325 = reinterpret_tensor(buf324, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf324  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_174], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_51.run(buf325, buf323, convolution_53, 5376, 2, grid=grid(5376), stream=stream0)
        del buf323
        buf326 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_33.run(buf325, buf326, 672, 8, grid=grid(672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf327 = aten.convolution_backward(buf325, mul_266, primals_191, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf325
        del mul_266
        del primals_191
        buf328 = buf327[0]
        buf329 = buf327[1]
        del buf327
        buf330 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_34.run(buf330, convolution_52, 224, grid=grid(224), stream=stream0)
        del convolution_52
        buf331 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_35.run(buf330, buf331, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf332 = aten.convolution_backward(buf330, mean_10, primals_189, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf330
        del mean_10
        del primals_189
        buf333 = buf332[0]
        buf334 = buf332[1]
        del buf332
        buf335 = reinterpret_tensor(buf309, (672, 9), (9, 1), 0); del buf309  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52.run(buf321, convolution_53, buf333, add_164, buf335, 6048, 128, grid=grid(6048), stream=stream0)
        buf336 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_42.run(buf335, buf336, 672, 9, grid=grid(672), stream=stream0)
        buf337 = reinterpret_tensor(buf335, (672, 9), (1, 672), 0); del buf335  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_53.run(buf321, convolution_53, buf333, add_164, convolution_51, unsqueeze_546, buf337, 6048, 128, grid=grid(6048), stream=stream0)
        buf338 = empty((672, ), device='cuda', dtype=torch.float32)
        buf340 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_44.run(buf337, squeeze_94, buf338, buf340, 672, 9, grid=grid(672), stream=stream0)
        buf339 = reinterpret_tensor(buf312, (8, 672, 12, 12), (96768, 1, 8064, 672), 0); del buf312  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_54.run(buf321, convolution_53, buf333, add_164, convolution_51, unsqueeze_546, buf338, squeeze_94, buf336, buf339, 1152, 672, grid=grid(1152, 672), stream=stream0)
        del add_164
        del buf333
        del convolution_51
        del convolution_53
        del unsqueeze_546
        buf341 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf339, squeeze_94, primals_63, buf341, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del buf339
        del primals_63
        del squeeze_94
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf342 = aten.convolution_backward(buf341, mul_257, primals_188, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
        del buf341
        del mul_257
        del primals_188
        buf343 = buf342[0]
        buf344 = buf342[1]
        del buf342
        buf345 = reinterpret_tensor(buf337, (672, 9), (9, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_41.run(buf343, mul_844, buf345, 6048, 128, grid=grid(6048), stream=stream0)
        buf346 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_42.run(buf345, buf346, 672, 9, grid=grid(672), stream=stream0)
        buf347 = reinterpret_tensor(buf345, (672, 9), (1, 672), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_43.run(buf343, mul_844, convolution_50, unsqueeze_558, buf347, 6048, 128, grid=grid(6048), stream=stream0)
        buf348 = empty((672, ), device='cuda', dtype=torch.float32)
        buf349 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_44.run(buf347, squeeze_91, buf348, buf349, 672, 9, grid=grid(672), stream=stream0)
        del buf347
        buf350 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_45.run(buf350, mul_844, convolution_50, unsqueeze_558, buf348, squeeze_91, buf346, primals_61, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del buf348
        del convolution_50
        del mul_844
        del primals_61
        del squeeze_91
        del unsqueeze_558
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf351 = aten.convolution_backward(buf350, add_154, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_154
        del buf350
        del primals_187
        buf352 = buf351[0]
        buf353 = buf351[1]
        del buf351
        buf354 = buf317; del buf317  # reuse
        buf355 = empty((112, ), device='cuda', dtype=torch.float32)
        buf357 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_60.run(buf237, buf276, buf314, buf352, convolution_49, unsqueeze_570, squeeze_88, buf354, buf355, buf357, 112, 1152, grid=grid(112), stream=stream0)
        buf356 = buf237; del buf237  # reuse
        buf358 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_61.run(buf358, buf276, buf314, buf352, convolution_49, unsqueeze_570, buf355, squeeze_88, buf354, primals_59, 896, 144, grid=grid(896, 144), stream=stream0)
        del buf276
        del buf314
        del buf352
        del buf355
        del convolution_49
        del primals_59
        del squeeze_88
        del unsqueeze_570
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf359 = aten.convolution_backward(buf358, mul_242, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf358
        del mul_242
        del primals_186
        buf360 = buf359[0]
        buf361 = buf359[1]
        del buf359
        buf362 = empty_strided((8, 480, 1, 1, 2), (960, 2, 7680, 7680, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_62.run(buf360, add_149, buf362, 7680, 72, grid=grid(7680), stream=stream0)
        buf363 = reinterpret_tensor(buf5, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf5  # reuse
        buf364 = reinterpret_tensor(buf363, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf363  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_158], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_63.run(buf364, buf362, convolution_48, 3840, 2, grid=grid(3840), stream=stream0)
        buf365 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_64.run(buf364, buf365, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf366 = aten.convolution_backward(buf364, mul_241, primals_184, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf364
        del mul_241
        del primals_184
        buf367 = buf366[0]
        buf368 = buf366[1]
        del buf366
        buf369 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_65.run(buf369, convolution_47, 160, grid=grid(160), stream=stream0)
        del convolution_47
        buf370 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_66.run(buf369, buf370, 20, 8, grid=grid(20), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf371 = aten.convolution_backward(buf369, mean_9, primals_182, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf369
        del mean_9
        del primals_182
        buf372 = buf371[0]
        buf373 = buf371[1]
        del buf371
        buf374 = empty((480, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf360, convolution_48, buf372, add_149, buf374, 4320, 128, grid=grid(4320), stream=stream0)
        buf375 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf374, buf375, 480, 9, grid=grid(480), stream=stream0)
        buf376 = reinterpret_tensor(buf374, (480, 9), (1, 480), 0); del buf374  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf360, convolution_48, buf372, add_149, convolution_46, unsqueeze_582, buf376, 4320, 128, grid=grid(4320), stream=stream0)
        buf377 = empty((480, ), device='cuda', dtype=torch.float32)
        buf379 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf376, squeeze_85, buf377, buf379, 480, 9, grid=grid(480), stream=stream0)
        buf378 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71.run(buf360, convolution_48, buf372, add_149, convolution_46, unsqueeze_582, buf377, squeeze_85, buf375, buf378, 1152, 480, grid=grid(1152, 480), stream=stream0)
        del add_149
        del convolution_46
        del convolution_48
        del unsqueeze_582
        buf380 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_72.run(buf378, squeeze_85, primals_57, buf380, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del buf378
        del primals_57
        del squeeze_85
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf381 = aten.convolution_backward(buf380, mul_232, primals_181, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False])
        del buf380
        del mul_232
        del primals_181
        buf382 = buf381[0]
        buf383 = buf381[1]
        del buf381
        buf384 = reinterpret_tensor(buf376, (480, 9), (9, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_73.run(buf382, mul_884, buf384, 4320, 128, grid=grid(4320), stream=stream0)
        buf385 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf384, buf385, 480, 9, grid=grid(480), stream=stream0)
        buf386 = reinterpret_tensor(buf384, (480, 9), (1, 480), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_74.run(buf382, mul_884, convolution_45, unsqueeze_594, buf386, 4320, 128, grid=grid(4320), stream=stream0)
        buf387 = empty((480, ), device='cuda', dtype=torch.float32)
        buf388 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf386, squeeze_82, buf387, buf388, 480, 9, grid=grid(480), stream=stream0)
        buf389 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_75.run(buf389, mul_884, convolution_45, unsqueeze_594, buf387, squeeze_82, buf385, primals_55, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del convolution_45
        del mul_884
        del primals_55
        del squeeze_82
        del unsqueeze_594
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf390 = aten.convolution_backward(buf389, add_139, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_139
        del primals_180
        buf391 = buf390[0]
        buf392 = buf390[1]
        del buf390
        buf393 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_76.run(buf391, buf393, 80, 1152, grid=grid(80), stream=stream0)
        buf394 = empty((80, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_77.run(buf391, convolution_44, unsqueeze_606, buf394, 720, 128, grid=grid(720), stream=stream0)
        buf395 = empty((80, ), device='cuda', dtype=torch.float32)
        buf396 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_78.run(buf394, squeeze_79, buf395, buf396, 80, 9, grid=grid(80), stream=stream0)
        del buf394
        buf397 = reinterpret_tensor(buf16, (8, 80, 12, 12), (11520, 144, 12, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_79.run(buf391, convolution_44, unsqueeze_606, buf395, squeeze_79, buf393, primals_53, buf397, 640, 144, grid=grid(640, 144), stream=stream0)
        del convolution_44
        del primals_53
        del squeeze_79
        del unsqueeze_606
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf398 = aten.convolution_backward(buf397, mul_217, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_217
        del primals_179
        buf399 = buf398[0]
        buf400 = buf398[1]
        del buf398
        buf401 = buf362; del buf362  # reuse
        # Source Nodes: [x_141], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_62.run(buf399, add_133, buf401, 7680, 72, grid=grid(7680), stream=stream0)
        buf402 = reinterpret_tensor(buf372, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf372  # reuse
        buf403 = reinterpret_tensor(buf402, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf402  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_141], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_63.run(buf403, buf401, convolution_43, 3840, 2, grid=grid(3840), stream=stream0)
        buf404 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_64.run(buf403, buf404, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf405 = aten.convolution_backward(buf403, mul_216, primals_177, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf403
        del mul_216
        del primals_177
        buf406 = buf405[0]
        buf407 = buf405[1]
        del buf405
        buf408 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_65.run(buf408, convolution_42, 160, grid=grid(160), stream=stream0)
        del convolution_42
        buf409 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_66.run(buf408, buf409, 20, 8, grid=grid(20), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf410 = aten.convolution_backward(buf408, mean_8, primals_175, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf408
        del mean_8
        del primals_175
        buf411 = buf410[0]
        buf412 = buf410[1]
        del buf410
        buf413 = reinterpret_tensor(buf386, (480, 9), (9, 1), 0); del buf386  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf399, convolution_43, buf411, add_133, buf413, 4320, 128, grid=grid(4320), stream=stream0)
        buf414 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf413, buf414, 480, 9, grid=grid(480), stream=stream0)
        buf415 = reinterpret_tensor(buf413, (480, 9), (1, 480), 0); del buf413  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf399, convolution_43, buf411, add_133, convolution_41, unsqueeze_618, buf415, 4320, 128, grid=grid(4320), stream=stream0)
        buf416 = empty((480, ), device='cuda', dtype=torch.float32)
        buf418 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf415, squeeze_76, buf416, buf418, 480, 9, grid=grid(480), stream=stream0)
        buf417 = reinterpret_tensor(buf389, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf389  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71.run(buf399, convolution_43, buf411, add_133, convolution_41, unsqueeze_618, buf416, squeeze_76, buf414, buf417, 1152, 480, grid=grid(1152, 480), stream=stream0)
        del add_133
        del convolution_41
        del convolution_43
        del unsqueeze_618
        buf419 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_72.run(buf417, squeeze_76, primals_51, buf419, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del buf417
        del primals_51
        del squeeze_76
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf420 = aten.convolution_backward(buf419, mul_207, primals_174, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
        del buf419
        del mul_207
        del primals_174
        buf421 = buf420[0]
        buf422 = buf420[1]
        del buf420
        buf423 = reinterpret_tensor(buf415, (480, 9), (9, 1), 0); del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_73.run(buf421, mul_924, buf423, 4320, 128, grid=grid(4320), stream=stream0)
        buf424 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf423, buf424, 480, 9, grid=grid(480), stream=stream0)
        buf425 = reinterpret_tensor(buf423, (480, 9), (1, 480), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_74.run(buf421, mul_924, convolution_40, unsqueeze_630, buf425, 4320, 128, grid=grid(4320), stream=stream0)
        buf426 = empty((480, ), device='cuda', dtype=torch.float32)
        buf427 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf425, squeeze_73, buf426, buf427, 480, 9, grid=grid(480), stream=stream0)
        buf428 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_75.run(buf428, mul_924, convolution_40, unsqueeze_630, buf426, squeeze_73, buf424, primals_49, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del convolution_40
        del mul_924
        del primals_49
        del squeeze_73
        del unsqueeze_630
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf429 = aten.convolution_backward(buf428, add_123, primals_173, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_123
        del primals_173
        buf430 = buf429[0]
        buf431 = buf429[1]
        del buf429
        buf432 = buf395; del buf395  # reuse
        buf433 = empty((80, ), device='cuda', dtype=torch.float32)
        buf434 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_80.run(buf391, buf430, convolution_39, unsqueeze_642, squeeze_70, buf432, buf433, buf434, 80, 1152, grid=grid(80), stream=stream0)
        buf435 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_81.run(buf391, buf430, convolution_39, unsqueeze_642, buf433, squeeze_70, buf432, primals_47, buf435, 640, 144, grid=grid(640, 144), stream=stream0)
        del convolution_39
        del primals_47
        del squeeze_70
        del unsqueeze_642
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf436 = aten.convolution_backward(buf435, mul_192, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_192
        del primals_172
        buf437 = buf436[0]
        buf438 = buf436[1]
        del buf436
        buf439 = buf401; del buf401  # reuse
        # Source Nodes: [x_124], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_62.run(buf437, add_117, buf439, 7680, 72, grid=grid(7680), stream=stream0)
        buf440 = reinterpret_tensor(buf411, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf411  # reuse
        buf441 = reinterpret_tensor(buf440, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf440  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_124], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_63.run(buf441, buf439, convolution_38, 3840, 2, grid=grid(3840), stream=stream0)
        buf442 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_64.run(buf441, buf442, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf443 = aten.convolution_backward(buf441, mul_191, primals_170, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf441
        del mul_191
        del primals_170
        buf444 = buf443[0]
        buf445 = buf443[1]
        del buf443
        buf446 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_65.run(buf446, convolution_37, 160, grid=grid(160), stream=stream0)
        del convolution_37
        buf447 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_66.run(buf446, buf447, 20, 8, grid=grid(20), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf448 = aten.convolution_backward(buf446, mean_7, primals_168, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf446
        del mean_7
        del primals_168
        buf449 = buf448[0]
        buf450 = buf448[1]
        del buf448
        buf451 = reinterpret_tensor(buf425, (480, 9), (9, 1), 0); del buf425  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf437, convolution_38, buf449, add_117, buf451, 4320, 128, grid=grid(4320), stream=stream0)
        buf452 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf451, buf452, 480, 9, grid=grid(480), stream=stream0)
        buf453 = reinterpret_tensor(buf451, (480, 9), (1, 480), 0); del buf451  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf437, convolution_38, buf449, add_117, convolution_36, unsqueeze_654, buf453, 4320, 128, grid=grid(4320), stream=stream0)
        buf454 = empty((480, ), device='cuda', dtype=torch.float32)
        buf456 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf453, squeeze_67, buf454, buf456, 480, 9, grid=grid(480), stream=stream0)
        buf455 = reinterpret_tensor(buf428, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf428  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71.run(buf437, convolution_38, buf449, add_117, convolution_36, unsqueeze_654, buf454, squeeze_67, buf452, buf455, 1152, 480, grid=grid(1152, 480), stream=stream0)
        del add_117
        del convolution_36
        del convolution_38
        del unsqueeze_654
        buf457 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_72.run(buf455, squeeze_67, primals_45, buf457, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del buf455
        del primals_45
        del squeeze_67
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf458 = aten.convolution_backward(buf457, mul_182, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
        del buf457
        del mul_182
        del primals_167
        buf459 = buf458[0]
        buf460 = buf458[1]
        del buf458
        buf461 = reinterpret_tensor(buf453, (480, 9), (9, 1), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_73.run(buf459, mul_964, buf461, 4320, 128, grid=grid(4320), stream=stream0)
        buf462 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf461, buf462, 480, 9, grid=grid(480), stream=stream0)
        buf463 = reinterpret_tensor(buf461, (480, 9), (1, 480), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_74.run(buf459, mul_964, convolution_35, unsqueeze_666, buf463, 4320, 128, grid=grid(4320), stream=stream0)
        buf464 = empty((480, ), device='cuda', dtype=torch.float32)
        buf465 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf463, squeeze_64, buf464, buf465, 480, 9, grid=grid(480), stream=stream0)
        buf466 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_75.run(buf466, mul_964, convolution_35, unsqueeze_666, buf464, squeeze_64, buf462, primals_43, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del convolution_35
        del mul_964
        del primals_43
        del squeeze_64
        del unsqueeze_666
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf467 = aten.convolution_backward(buf466, add_107, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_107
        del primals_166
        buf468 = buf467[0]
        buf469 = buf467[1]
        del buf467
        buf470 = buf433; del buf433  # reuse
        buf471 = empty((80, ), device='cuda', dtype=torch.float32)
        buf473 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_82.run(buf391, buf430, buf468, convolution_34, unsqueeze_678, squeeze_61, buf470, buf471, buf473, 80, 1152, grid=grid(80), stream=stream0)
        buf472 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_83.run(buf391, buf430, buf468, convolution_34, unsqueeze_678, buf471, squeeze_61, buf470, primals_41, buf472, 640, 144, grid=grid(640, 144), stream=stream0)
        del convolution_34
        del primals_41
        del squeeze_61
        del unsqueeze_678
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf474 = aten.convolution_backward(buf472, mul_167, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf472
        del mul_167
        del primals_165
        buf475 = buf474[0]
        buf476 = buf474[1]
        del buf474
        buf477 = buf439; del buf439  # reuse
        # Source Nodes: [x_107], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_62.run(buf475, add_101, buf477, 7680, 72, grid=grid(7680), stream=stream0)
        buf478 = reinterpret_tensor(buf449, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf449  # reuse
        buf479 = reinterpret_tensor(buf478, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf478  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_63.run(buf479, buf477, convolution_33, 3840, 2, grid=grid(3840), stream=stream0)
        del buf477
        buf480 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_64.run(buf479, buf480, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf481 = aten.convolution_backward(buf479, mul_166, primals_163, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf479
        del mul_166
        del primals_163
        buf482 = buf481[0]
        buf483 = buf481[1]
        del buf481
        buf484 = buf482; del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_65.run(buf484, convolution_32, 160, grid=grid(160), stream=stream0)
        del convolution_32
        buf485 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_66.run(buf484, buf485, 20, 8, grid=grid(20), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf486 = aten.convolution_backward(buf484, mean_6, primals_161, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf484
        del mean_6
        del primals_161
        buf487 = buf486[0]
        buf488 = buf486[1]
        del buf486
        buf489 = reinterpret_tensor(buf463, (480, 9), (9, 1), 0); del buf463  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf475, convolution_33, buf487, add_101, buf489, 4320, 128, grid=grid(4320), stream=stream0)
        buf490 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf489, buf490, 480, 9, grid=grid(480), stream=stream0)
        buf491 = reinterpret_tensor(buf489, (480, 9), (1, 480), 0); del buf489  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf475, convolution_33, buf487, add_101, convolution_31, unsqueeze_690, buf491, 4320, 128, grid=grid(4320), stream=stream0)
        buf492 = empty((480, ), device='cuda', dtype=torch.float32)
        buf494 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf491, squeeze_58, buf492, buf494, 480, 9, grid=grid(480), stream=stream0)
        buf493 = reinterpret_tensor(buf466, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf466  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71.run(buf475, convolution_33, buf487, add_101, convolution_31, unsqueeze_690, buf492, squeeze_58, buf490, buf493, 1152, 480, grid=grid(1152, 480), stream=stream0)
        del add_101
        del convolution_31
        del convolution_33
        del unsqueeze_690
        buf495 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_72.run(buf493, squeeze_58, primals_39, buf495, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del buf493
        del primals_39
        del squeeze_58
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf496 = aten.convolution_backward(buf495, mul_157, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
        del buf495
        del mul_157
        del primals_160
        buf497 = buf496[0]
        buf498 = buf496[1]
        del buf496
        buf499 = reinterpret_tensor(buf491, (480, 9), (9, 1), 0); del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_73.run(buf497, mul_1004, buf499, 4320, 128, grid=grid(4320), stream=stream0)
        buf500 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf499, buf500, 480, 9, grid=grid(480), stream=stream0)
        buf501 = reinterpret_tensor(buf499, (480, 9), (1, 480), 0); del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_74.run(buf497, mul_1004, convolution_30, unsqueeze_702, buf501, 4320, 128, grid=grid(4320), stream=stream0)
        buf502 = empty((480, ), device='cuda', dtype=torch.float32)
        buf503 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf501, squeeze_55, buf502, buf503, 480, 9, grid=grid(480), stream=stream0)
        del buf501
        buf504 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_75.run(buf504, mul_1004, convolution_30, unsqueeze_702, buf502, squeeze_55, buf500, primals_37, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del buf502
        del convolution_30
        del mul_1004
        del primals_37
        del squeeze_55
        del unsqueeze_702
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf505 = aten.convolution_backward(buf504, add_91, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_91
        del buf504
        del primals_159
        buf506 = buf505[0]
        buf507 = buf505[1]
        del buf505
        buf508 = buf471; del buf471  # reuse
        buf509 = empty((80, ), device='cuda', dtype=torch.float32)
        buf511 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_84.run(buf391, buf430, buf468, buf506, convolution_29, unsqueeze_714, squeeze_52, buf508, buf509, buf511, 80, 1152, grid=grid(80), stream=stream0)
        buf510 = buf391; del buf391  # reuse
        buf512 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_85.run(buf512, buf430, buf468, buf506, convolution_29, unsqueeze_714, buf509, squeeze_52, buf508, primals_35, 640, 144, grid=grid(640, 144), stream=stream0)
        del buf430
        del buf468
        del buf506
        del buf509
        del convolution_29
        del primals_35
        del squeeze_52
        del unsqueeze_714
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf513 = aten.convolution_backward(buf512, mul_142, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf512
        del mul_142
        del primals_158
        buf514 = buf513[0]
        buf515 = buf513[1]
        del buf513
        buf516 = reinterpret_tensor(buf487, (8, 240, 1, 1, 2), (480, 2, 3840, 3840, 1), 0); del buf487  # reuse
        # Source Nodes: [x_91], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_86.run(buf514, add_86, buf516, 3840, 72, grid=grid(3840), stream=stream0)
        buf517 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf518 = reinterpret_tensor(buf517, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf517  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_87.run(buf518, buf516, convolution_28, 1920, 2, grid=grid(1920), stream=stream0)
        del buf516
        buf519 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_88.run(buf518, buf519, 240, 8, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf520 = aten.convolution_backward(buf518, mul_141, primals_156, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf518
        del mul_141
        del primals_156
        buf521 = buf520[0]
        buf522 = buf520[1]
        del buf520
        buf523 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_89.run(buf523, convolution_27, 80, grid=grid(80), stream=stream0)
        del convolution_27
        buf524 = empty((10, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_90.run(buf523, buf524, 10, 8, grid=grid(10), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf525 = aten.convolution_backward(buf523, mean_5, primals_154, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf523
        del mean_5
        del primals_154
        buf526 = buf525[0]
        buf527 = buf525[1]
        del buf525
        buf528 = empty((240, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_91.run(buf514, convolution_28, buf526, add_86, buf528, 2160, 128, grid=grid(2160), stream=stream0)
        buf529 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92.run(buf528, buf529, 240, 9, grid=grid(240), stream=stream0)
        buf530 = reinterpret_tensor(buf528, (240, 9), (1, 240), 0); del buf528  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_93.run(buf514, convolution_28, buf526, add_86, convolution_26, unsqueeze_726, buf530, 2160, 128, grid=grid(2160), stream=stream0)
        buf531 = empty((240, ), device='cuda', dtype=torch.float32)
        buf533 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_94.run(buf530, squeeze_49, buf531, buf533, 240, 9, grid=grid(240), stream=stream0)
        del buf530
        buf532 = empty_strided((8, 240, 12, 12), (34560, 1, 2880, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_95.run(buf514, convolution_28, buf526, add_86, convolution_26, unsqueeze_726, buf531, squeeze_49, buf529, buf532, 1152, 240, grid=grid(1152, 240), stream=stream0)
        del add_86
        del convolution_26
        del convolution_28
        del unsqueeze_726
        buf534 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_96.run(buf532, squeeze_49, primals_33, buf534, 1920, 144, grid=grid(1920, 144), stream=stream0)
        del buf532
        del primals_33
        del squeeze_49
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf535 = aten.convolution_backward(buf534, mul_132, primals_153, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False])
        del buf534
        del mul_132
        del primals_153
        buf536 = buf535[0]
        buf537 = buf535[1]
        del buf535
        buf538 = empty((240, 36), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_97.run(buf536, mul_1044, buf538, 8640, 128, grid=grid(8640), stream=stream0)
        buf539 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_98.run(buf538, buf539, 240, 36, grid=grid(240), stream=stream0)
        buf540 = reinterpret_tensor(buf538, (240, 36), (1, 240), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_99.run(buf536, mul_1044, convolution_25, unsqueeze_738, buf540, 8640, 128, grid=grid(8640), stream=stream0)
        buf541 = empty((240, ), device='cuda', dtype=torch.float32)
        buf542 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_100.run(buf540, squeeze_46, buf541, buf542, 240, 36, grid=grid(240), stream=stream0)
        buf543 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_101.run(buf543, mul_1044, convolution_25, unsqueeze_738, buf541, squeeze_46, buf539, primals_31, 1920, 576, grid=grid(1920, 576), stream=stream0)
        del convolution_25
        del mul_1044
        del primals_31
        del squeeze_46
        del unsqueeze_738
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf544 = aten.convolution_backward(buf543, add_76, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_76
        del primals_152
        buf545 = buf544[0]
        buf546 = buf544[1]
        del buf544
        buf547 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_102.run(buf545, buf547, 40, 4608, grid=grid(40), stream=stream0)
        buf548 = empty((40, 36), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_103.run(buf545, convolution_24, unsqueeze_750, buf548, 1440, 128, grid=grid(1440), stream=stream0)
        buf549 = empty((40, ), device='cuda', dtype=torch.float32)
        buf550 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_104.run(buf548, squeeze_43, buf549, buf550, 40, 36, grid=grid(40), stream=stream0)
        del buf548
        buf551 = empty((8, 40, 24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_105.run(buf545, convolution_24, unsqueeze_750, buf549, squeeze_43, buf547, primals_29, buf551, 320, 576, grid=grid(320, 576), stream=stream0)
        del convolution_24
        del primals_29
        del squeeze_43
        del unsqueeze_750
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf552 = aten.convolution_backward(buf551, mul_117, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf551
        del mul_117
        del primals_151
        buf553 = buf552[0]
        buf554 = buf552[1]
        del buf552
        buf555 = empty_strided((8, 240, 1, 1, 5), (1200, 5, 9600, 9600, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_106.run(buf553, add_70, buf555, 9600, 116, grid=grid(9600), stream=stream0)
        buf556 = reinterpret_tensor(buf526, (8, 240, 1, 1), (240, 1, 1920, 1920), 0); del buf526  # reuse
        buf557 = reinterpret_tensor(buf556, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf556  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_107.run(buf557, buf555, convolution_23, 1920, 5, grid=grid(1920), stream=stream0)
        del buf555
        buf558 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_88.run(buf557, buf558, 240, 8, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf559 = aten.convolution_backward(buf557, mul_116, primals_149, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf557
        del mul_116
        del primals_149
        buf560 = buf559[0]
        buf561 = buf559[1]
        del buf559
        buf562 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_89.run(buf562, convolution_22, 80, grid=grid(80), stream=stream0)
        del convolution_22
        buf563 = empty((10, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_90.run(buf562, buf563, 10, 8, grid=grid(10), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf564 = aten.convolution_backward(buf562, mean_4, primals_147, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf562
        del mean_4
        del primals_147
        buf565 = buf564[0]
        buf566 = buf564[1]
        del buf564
        buf567 = reinterpret_tensor(buf540, (240, 36), (36, 1), 0); del buf540  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_108.run(buf553, convolution_23, buf565, add_70, buf567, 8640, 128, grid=grid(8640), stream=stream0)
        buf568 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_98.run(buf567, buf568, 240, 36, grid=grid(240), stream=stream0)
        buf569 = reinterpret_tensor(buf567, (240, 36), (1, 240), 0); del buf567  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_109.run(buf553, convolution_23, buf565, add_70, convolution_21, unsqueeze_762, buf569, 8640, 128, grid=grid(8640), stream=stream0)
        buf570 = empty((240, ), device='cuda', dtype=torch.float32)
        buf572 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_100.run(buf569, squeeze_40, buf570, buf572, 240, 36, grid=grid(240), stream=stream0)
        buf571 = reinterpret_tensor(buf543, (8, 240, 24, 24), (138240, 1, 5760, 240), 0); del buf543  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_110.run(buf553, convolution_23, buf565, add_70, convolution_21, unsqueeze_762, buf570, squeeze_40, buf568, buf571, 4608, 240, grid=grid(4608, 240), stream=stream0)
        del add_70
        del buf565
        del convolution_21
        del convolution_23
        del unsqueeze_762
        buf573 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_111.run(buf571, squeeze_40, primals_27, buf573, 1920, 576, grid=grid(1920, 576), stream=stream0)
        del buf571
        del primals_27
        del squeeze_40
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf574 = aten.convolution_backward(buf573, mul_107, primals_146, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False])
        del buf573
        del mul_107
        del primals_146
        buf575 = buf574[0]
        buf576 = buf574[1]
        del buf574
        buf577 = reinterpret_tensor(buf569, (240, 36), (36, 1), 0); del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_97.run(buf575, mul_1084, buf577, 8640, 128, grid=grid(8640), stream=stream0)
        buf578 = buf570; del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_98.run(buf577, buf578, 240, 36, grid=grid(240), stream=stream0)
        buf579 = reinterpret_tensor(buf577, (240, 36), (1, 240), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_99.run(buf575, mul_1084, convolution_20, unsqueeze_774, buf579, 8640, 128, grid=grid(8640), stream=stream0)
        buf580 = empty((240, ), device='cuda', dtype=torch.float32)
        buf581 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_100.run(buf579, squeeze_37, buf580, buf581, 240, 36, grid=grid(240), stream=stream0)
        del buf579
        buf582 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_101.run(buf582, mul_1084, convolution_20, unsqueeze_774, buf580, squeeze_37, buf578, primals_25, 1920, 576, grid=grid(1920, 576), stream=stream0)
        del buf580
        del convolution_20
        del mul_1084
        del primals_25
        del squeeze_37
        del unsqueeze_774
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf583 = aten.convolution_backward(buf582, add_60, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_60
        del buf582
        del primals_145
        buf584 = buf583[0]
        buf585 = buf583[1]
        del buf583
        buf586 = buf549; del buf549  # reuse
        buf587 = empty((40, ), device='cuda', dtype=torch.float32)
        buf588 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_112.run(buf545, buf584, convolution_19, unsqueeze_786, squeeze_34, buf586, buf587, buf588, 40, 4608, grid=grid(40), stream=stream0)
        buf589 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_113.run(buf589, buf584, convolution_19, unsqueeze_786, buf587, squeeze_34, buf586, primals_23, 320, 576, grid=grid(320, 576), stream=stream0)
        del buf584
        del buf587
        del convolution_19
        del primals_23
        del squeeze_34
        del unsqueeze_786
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf590 = aten.convolution_backward(buf589, mul_92, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf589
        del mul_92
        del primals_144
        buf591 = buf590[0]
        buf592 = buf590[1]
        del buf590
        buf593 = empty_strided((8, 144, 1, 1, 5), (720, 5, 5760, 5760, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_58], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_114.run(buf591, add_55, buf593, 5760, 116, grid=grid(5760), stream=stream0)
        buf594 = reinterpret_tensor(buf194, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf194  # reuse
        buf595 = reinterpret_tensor(buf594, (8, 144, 1, 1), (144, 1, 1, 1), 0); del buf594  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_115.run(buf595, buf593, convolution_18, 1152, 5, grid=grid(1152), stream=stream0)
        del buf593
        buf596 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_116.run(buf595, buf596, 144, 8, grid=grid(144), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf597 = aten.convolution_backward(buf595, mul_91, primals_142, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf595
        del mul_91
        del primals_142
        buf598 = buf597[0]
        buf599 = buf597[1]
        del buf597
        buf600 = buf598; del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_117.run(buf600, convolution_17, 48, grid=grid(48), stream=stream0)
        del convolution_17
        buf601 = empty((6, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_118.run(buf600, buf601, 6, 8, grid=grid(6), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf602 = aten.convolution_backward(buf600, mean_3, primals_140, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf600
        del mean_3
        del primals_140
        buf603 = buf602[0]
        buf604 = buf602[1]
        del buf602
        buf605 = empty((144, 36), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_119.run(buf591, convolution_18, buf603, add_55, buf605, 5184, 128, grid=grid(5184), stream=stream0)
        buf606 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_120.run(buf605, buf606, 144, 36, grid=grid(144), stream=stream0)
        buf607 = reinterpret_tensor(buf605, (144, 36), (1, 144), 0); del buf605  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_121.run(buf591, convolution_18, buf603, add_55, convolution_16, unsqueeze_798, buf607, 5184, 128, grid=grid(5184), stream=stream0)
        buf608 = empty((144, ), device='cuda', dtype=torch.float32)
        buf610 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_122.run(buf607, squeeze_31, buf608, buf610, 144, 36, grid=grid(144), stream=stream0)
        del buf607
        buf609 = empty_strided((8, 144, 24, 24), (82944, 1, 3456, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_123.run(buf591, convolution_18, buf603, add_55, convolution_16, unsqueeze_798, buf608, squeeze_31, buf606, buf609, 4608, 144, grid=grid(4608, 144), stream=stream0)
        del add_55
        del convolution_16
        del convolution_18
        del unsqueeze_798
        buf611 = buf591; del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_124.run(buf609, squeeze_31, primals_21, buf611, 1152, 576, grid=grid(1152, 576), stream=stream0)
        del buf609
        del primals_21
        del squeeze_31
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf612 = aten.convolution_backward(buf611, mul_82, primals_139, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 144, [True, True, False])
        del buf611
        del mul_82
        del primals_139
        buf613 = buf612[0]
        buf614 = buf612[1]
        del buf612
        buf615 = empty((144, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_125.run(buf613, mul_1124, buf615, 20736, 128, grid=grid(20736), stream=stream0)
        buf616 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_126.run(buf615, buf616, 144, 144, grid=grid(144), stream=stream0)
        buf617 = reinterpret_tensor(buf615, (144, 144), (1, 144), 0); del buf615  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_127.run(buf613, mul_1124, convolution_15, unsqueeze_810, buf617, 20736, 128, grid=grid(20736), stream=stream0)
        buf618 = empty((144, ), device='cuda', dtype=torch.float32)
        buf619 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_128.run(buf617, squeeze_28, buf618, buf619, 144, 144, grid=grid(144), stream=stream0)
        buf620 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_129.run(buf620, mul_1124, convolution_15, unsqueeze_810, buf618, squeeze_28, buf616, primals_19, 1152, 2304, grid=grid(1152, 2304), stream=stream0)
        del convolution_15
        del mul_1124
        del primals_19
        del squeeze_28
        del unsqueeze_810
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf621 = aten.convolution_backward(buf620, add_45, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_45
        del primals_138
        buf622 = buf621[0]
        buf623 = buf621[1]
        del buf621
        buf624 = empty_strided((24, 3), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_130.run(buf622, buf624, 72, 6144, grid=grid(72), stream=stream0)
        buf625 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_131.run(buf624, buf625, 24, 3, grid=grid(24), stream=stream0)
        buf626 = reinterpret_tensor(buf193, (24, 144), (144, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_132.run(buf622, convolution_14, unsqueeze_822, buf626, 3456, 128, grid=grid(3456), stream=stream0)
        buf627 = empty((24, ), device='cuda', dtype=torch.float32)
        buf628 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_133.run(buf626, squeeze_25, buf627, buf628, 24, 144, grid=grid(24), stream=stream0)
        del buf626
        buf629 = empty((8, 24, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_134.run(buf622, convolution_14, unsqueeze_822, buf627, squeeze_25, buf625, primals_17, buf629, 192, 2304, grid=grid(192, 2304), stream=stream0)
        del convolution_14
        del primals_17
        del squeeze_25
        del unsqueeze_822
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf630 = aten.convolution_backward(buf629, mul_67, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf629
        del mul_67
        del primals_137
        buf631 = buf630[0]
        buf632 = buf630[1]
        del buf630
        buf633 = reinterpret_tensor(buf617, (8, 144, 1, 1, 18), (2592, 18, 20736, 20736, 1), 0); del buf617  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_135.run(buf631, add_39, buf633, 20736, 128, grid=grid(20736), stream=stream0)
        buf634 = reinterpret_tensor(buf603, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf603  # reuse
        buf635 = reinterpret_tensor(buf634, (8, 144, 1, 1), (144, 1, 1, 1), 0); del buf634  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_136.run(buf635, buf633, convolution_13, 1152, 18, grid=grid(1152), stream=stream0)
        buf636 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_116.run(buf635, buf636, 144, 8, grid=grid(144), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf637 = aten.convolution_backward(buf635, mul_66, primals_135, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf635
        del mul_66
        del primals_135
        buf638 = buf637[0]
        buf639 = buf637[1]
        del buf637
        buf640 = buf638; del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_117.run(buf640, convolution_12, 48, grid=grid(48), stream=stream0)
        del convolution_12
        buf641 = empty((6, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_118.run(buf640, buf641, 6, 8, grid=grid(6), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf642 = aten.convolution_backward(buf640, mean_2, primals_133, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf640
        del mean_2
        del primals_133
        buf643 = buf642[0]
        buf644 = buf642[1]
        del buf642
        buf645 = reinterpret_tensor(buf633, (144, 144), (144, 1), 0); del buf633  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_137.run(buf631, convolution_13, buf643, add_39, buf645, 20736, 128, grid=grid(20736), stream=stream0)
        buf646 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_126.run(buf645, buf646, 144, 144, grid=grid(144), stream=stream0)
        buf647 = reinterpret_tensor(buf645, (144, 144), (1, 144), 0); del buf645  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_138.run(buf631, convolution_13, buf643, add_39, convolution_11, unsqueeze_834, buf647, 20736, 128, grid=grid(20736), stream=stream0)
        buf648 = empty((144, ), device='cuda', dtype=torch.float32)
        buf650 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_mul_native_batch_norm_backward_128.run(buf647, squeeze_22, buf648, buf650, 144, 144, grid=grid(144), stream=stream0)
        buf649 = reinterpret_tensor(buf620, (8, 144, 48, 48), (331776, 1, 6912, 144), 0); del buf620  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_139.run(buf631, convolution_13, buf643, add_39, convolution_11, unsqueeze_834, buf648, squeeze_22, buf646, buf649, 18432, 144, grid=grid(18432, 144), stream=stream0)
        del add_39
        del buf643
        del convolution_11
        del convolution_13
        del unsqueeze_834
        buf651 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_140.run(buf649, squeeze_22, primals_15, buf651, 1152, 2304, grid=grid(1152, 2304), stream=stream0)
        del buf649
        del primals_15
        del squeeze_22
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf652 = aten.convolution_backward(buf651, mul_57, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
        del buf651
        del mul_57
        del primals_132
        buf653 = buf652[0]
        buf654 = buf652[1]
        del buf652
        buf655 = reinterpret_tensor(buf647, (144, 144), (144, 1), 0); del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_125.run(buf653, mul_1164, buf655, 20736, 128, grid=grid(20736), stream=stream0)
        buf656 = buf648; del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_126.run(buf655, buf656, 144, 144, grid=grid(144), stream=stream0)
        buf657 = reinterpret_tensor(buf655, (144, 144), (1, 144), 0); del buf655  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_127.run(buf653, mul_1164, convolution_10, unsqueeze_846, buf657, 20736, 128, grid=grid(20736), stream=stream0)
        buf658 = empty((144, ), device='cuda', dtype=torch.float32)
        buf659 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_128.run(buf657, squeeze_19, buf658, buf659, 144, 144, grid=grid(144), stream=stream0)
        del buf657
        buf660 = buf653; del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_129.run(buf660, mul_1164, convolution_10, unsqueeze_846, buf658, squeeze_19, buf656, primals_13, 1152, 2304, grid=grid(1152, 2304), stream=stream0)
        del convolution_10
        del mul_1164
        del primals_13
        del squeeze_19
        del unsqueeze_846
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf661 = aten.convolution_backward(buf660, add_29, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_29
        del buf660
        del primals_131
        buf662 = buf661[0]
        buf663 = buf661[1]
        del buf661
        buf664 = buf624; del buf624  # reuse
        buf666 = empty_strided((24, 3), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_141.run(buf622, buf662, convolution_9, unsqueeze_858, buf664, buf666, 72, 6144, grid=grid(72), stream=stream0)
        buf665 = buf627; del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_131.run(buf664, buf665, 24, 3, grid=grid(24), stream=stream0)
        del buf664
        buf667 = empty((24, ), device='cuda', dtype=torch.float32)
        buf668 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_142.run(buf666, squeeze_16, buf667, buf668, 24, 3, grid=grid(24), stream=stream0)
        del buf666
        buf669 = buf622; del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_143.run(buf669, buf662, convolution_9, unsqueeze_858, buf667, squeeze_16, buf665, primals_11, 192, 2304, grid=grid(192, 2304), stream=stream0)
        del buf662
        del buf667
        del convolution_9
        del primals_11
        del squeeze_16
        del unsqueeze_858
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf670 = aten.convolution_backward(buf669, mul_42, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf669
        del mul_42
        del primals_130
        buf671 = buf670[0]
        buf672 = buf670[1]
        del buf670
        buf673 = empty_strided((8, 96, 1, 1, 18), (1728, 18, 13824, 13824, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_144.run(buf671, add_24, buf673, 13824, 128, grid=grid(13824), stream=stream0)
        buf674 = empty_strided((8, 96, 1, 1), (96, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf675 = reinterpret_tensor(buf674, (8, 96, 1, 1), (96, 1, 1, 1), 0); del buf674  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_145.run(buf675, buf673, convolution_8, 768, 18, grid=grid(768), stream=stream0)
        buf676 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_146.run(buf675, buf676, 96, 8, grid=grid(96), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf677 = aten.convolution_backward(buf675, mul_41, primals_128, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf675
        del mul_41
        del primals_128
        buf678 = buf677[0]
        buf679 = buf677[1]
        del buf677
        buf680 = buf678; del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_147.run(buf680, convolution_7, 32, grid=grid(32), stream=stream0)
        del convolution_7
        buf681 = empty((4, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_148.run(buf680, buf681, 4, 8, grid=grid(4), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf682 = aten.convolution_backward(buf680, mean_1, primals_126, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_1
        del primals_126
        buf683 = buf682[0]
        buf684 = buf682[1]
        del buf682
        buf685 = reinterpret_tensor(buf673, (96, 144), (144, 1), 0); del buf673  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_149.run(buf671, convolution_8, buf683, add_24, buf685, 13824, 128, grid=grid(13824), stream=stream0)
        buf686 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_150.run(buf685, buf686, 96, 144, grid=grid(96), stream=stream0)
        buf687 = reinterpret_tensor(buf685, (96, 144), (1, 96), 0); del buf685  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_151.run(buf671, convolution_8, buf683, add_24, convolution_6, unsqueeze_870, buf687, 13824, 128, grid=grid(13824), stream=stream0)
        buf688 = empty((96, ), device='cuda', dtype=torch.float32)
        buf690 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_152.run(buf687, squeeze_13, buf688, buf690, 96, 144, grid=grid(96), stream=stream0)
        del buf687
        buf689 = empty_strided((8, 96, 48, 48), (221184, 1, 4608, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_153.run(buf671, convolution_8, buf683, add_24, convolution_6, unsqueeze_870, buf688, squeeze_13, buf686, buf689, 18432, 96, grid=grid(18432, 96), stream=stream0)
        del add_24
        del buf683
        del convolution_6
        del convolution_8
        del unsqueeze_870
        buf691 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_154.run(buf689, squeeze_13, primals_9, buf691, 768, 2304, grid=grid(768, 2304), stream=stream0)
        del buf689
        del primals_9
        del squeeze_13
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf692 = aten.convolution_backward(buf691, mul_32, primals_125, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False])
        del buf691
        del mul_32
        del primals_125
        buf693 = buf692[0]
        buf694 = buf692[1]
        del buf692
        buf695 = reinterpret_tensor(buf205, (96, 576), (576, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_155.run(buf693, mul_1204, buf695, 55296, 128, grid=grid(55296), stream=stream0)
        buf696 = buf688; del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_156.run(buf695, buf696, 96, 576, grid=grid(96), stream=stream0)
        buf697 = reinterpret_tensor(buf695, (96, 576), (1, 96), 0); del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_157.run(buf693, mul_1204, convolution_5, unsqueeze_882, buf697, 55296, 128, grid=grid(55296), stream=stream0)
        buf698 = empty((96, ), device='cuda', dtype=torch.float32)
        buf699 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_158.run(buf697, squeeze_10, buf698, buf699, 96, 576, grid=grid(96), stream=stream0)
        del buf697
        buf700 = buf693; del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_159.run(buf700, mul_1204, convolution_5, unsqueeze_882, buf698, squeeze_10, buf696, primals_7, 768, 9216, grid=grid(768, 9216), stream=stream0)
        del buf698
        del convolution_5
        del mul_1204
        del primals_7
        del squeeze_10
        del unsqueeze_882
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf701 = aten.convolution_backward(buf700, add_14, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_14
        del buf700
        del primals_124
        buf702 = buf701[0]
        buf703 = buf701[1]
        del buf701
        buf704 = reinterpret_tensor(buf658, (16, 9), (9, 1), 0); del buf658  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_160.run(buf702, buf704, 144, 8192, grid=grid(144), stream=stream0)
        buf705 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_161.run(buf704, buf705, 16, 9, grid=grid(16), stream=stream0)
        del buf704
        buf706 = reinterpret_tensor(buf179, (16, 576), (576, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_162.run(buf702, convolution_4, unsqueeze_894, buf706, 9216, 128, grid=grid(9216), stream=stream0)
        buf707 = empty((16, ), device='cuda', dtype=torch.float32)
        buf708 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_163.run(buf706, squeeze_7, buf707, buf708, 16, 576, grid=grid(16), stream=stream0)
        del buf706
        buf709 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_164.run(buf709, convolution_4, unsqueeze_894, buf707, squeeze_7, buf705, primals_5, 128, 9216, grid=grid(128, 9216), stream=stream0)
        del buf707
        del convolution_4
        del primals_5
        del squeeze_7
        del unsqueeze_894
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf710 = aten.convolution_backward(buf709, mul_17, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf709
        del mul_17
        del primals_123
        buf711 = buf710[0]
        buf712 = buf710[1]
        del buf710
        buf713 = empty_strided((8, 32, 1, 1, 72), (2304, 72, 18432, 18432, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_165.run(buf711, add_9, buf713, 18432, 128, grid=grid(18432), stream=stream0)
        buf714 = empty_strided((8, 32, 1, 1), (32, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf715 = reinterpret_tensor(buf714, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf714  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_9], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_166.run(buf715, buf713, convolution_3, 256, 72, grid=grid(256), stream=stream0)
        buf716 = reinterpret_tensor(buf680, (32, ), (1, ), 0); del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_167.run(buf715, buf716, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf717 = aten.convolution_backward(buf715, mul_16, primals_121, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf715
        del mul_16
        del primals_121
        buf718 = buf717[0]
        buf719 = buf717[1]
        del buf717
        buf720 = buf718; del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_168.run(buf720, convolution_2, 64, grid=grid(64), stream=stream0)
        del convolution_2
        buf721 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_169.run(buf720, buf721, 8, 8, grid=grid(8), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf722 = aten.convolution_backward(buf720, mean, primals_119, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf720
        del mean
        del primals_119
        buf723 = buf722[0]
        buf724 = buf722[1]
        del buf722
        buf725 = reinterpret_tensor(buf713, (32, 576), (576, 1), 0); del buf713  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_170.run(buf711, convolution_3, buf723, add_9, buf725, 18432, 128, grid=grid(18432), stream=stream0)
        buf726 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_171.run(buf725, buf726, 32, 576, grid=grid(32), stream=stream0)
        buf727 = reinterpret_tensor(buf725, (32, 576), (1, 32), 0); del buf725  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_172.run(buf711, convolution_3, buf723, add_9, convolution_1, unsqueeze_906, buf727, 18432, 128, grid=grid(18432), stream=stream0)
        buf728 = empty((32, ), device='cuda', dtype=torch.float32)
        buf730 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_173.run(buf727, squeeze_4, buf728, buf730, 32, 576, grid=grid(32), stream=stream0)
        buf729 = empty_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_174.run(buf711, convolution_3, buf723, add_9, convolution_1, unsqueeze_906, buf728, squeeze_4, buf726, buf729, 73728, 32, grid=grid(73728, 32), stream=stream0)
        del add_9
        del buf723
        del convolution_1
        del convolution_3
        del unsqueeze_906
        buf731 = buf711; del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_175.run(buf729, squeeze_4, primals_3, buf731, 256, 9216, grid=grid(256, 9216), stream=stream0)
        del buf729
        del primals_3
        del squeeze_4
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf732 = aten.convolution_backward(buf731, mul_7, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf731
        del mul_7
        del primals_118
        buf733 = buf732[0]
        buf734 = buf732[1]
        del buf732
        buf735 = reinterpret_tensor(buf727, (32, 576), (576, 1), 0); del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_176.run(buf733, mul_1244, buf735, 18432, 128, grid=grid(18432), stream=stream0)
        buf736 = buf728; del buf728  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_171.run(buf735, buf736, 32, 576, grid=grid(32), stream=stream0)
        buf737 = reinterpret_tensor(buf735, (32, 576), (1, 32), 0); del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_177.run(buf733, mul_1244, convolution, unsqueeze_918, buf737, 18432, 128, grid=grid(18432), stream=stream0)
        buf738 = empty((32, ), device='cuda', dtype=torch.float32)
        buf739 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_173.run(buf737, squeeze_1, buf738, buf739, 32, 576, grid=grid(32), stream=stream0)
        del buf737
        buf740 = buf733; del buf733  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_178.run(buf740, mul_1244, convolution, unsqueeze_918, buf738, squeeze_1, buf736, primals_1, 256, 9216, grid=grid(256, 9216), stream=stream0)
        del buf738
        del convolution
        del mul_1244
        del primals_1
        del squeeze_1
        del unsqueeze_918
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf741 = aten.convolution_backward(buf740, primals_427, primals_117, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf740
        del primals_117
        del primals_427
        buf742 = buf741[1]
        return (buf739, buf736, buf730, buf726, buf708, buf705, buf699, buf696, buf690, buf686, buf668, buf665, buf659, buf656, buf650, buf646, buf628, buf625, buf619, buf616, buf610, buf606, buf588, buf586, buf581, buf578, buf572, buf568, buf550, buf547, buf542, buf539, buf533, buf529, buf511, buf508, buf503, buf500, buf494, buf490, buf473, buf470, buf465, buf462, buf456, buf452, buf434, buf432, buf427, buf424, buf418, buf414, buf396, buf393, buf388, buf385, buf379, buf375, buf357, buf354, buf349, buf346, buf340, buf336, buf319, buf316, buf311, buf308, buf302, buf298, buf280, buf278, buf273, buf270, buf264, buf260, buf242, buf239, buf234, buf231, buf225, buf221, buf204, buf201, buf195, buf192, buf186, buf182, buf165, buf162, buf157, buf154, buf148, buf144, buf128, buf125, buf120, buf117, buf111, buf107, buf90, buf88, buf83, buf80, buf74, buf70, buf53, buf50, buf45, buf42, buf36, buf32, buf15, buf12, buf7, buf4, buf742, buf734, buf724, buf721, buf719, buf716, buf712, buf703, buf694, buf684, buf681, buf679, buf676, buf672, buf663, buf654, buf644, buf641, buf639, buf636, buf632, buf623, buf614, buf604, buf601, buf599, buf596, buf592, buf585, buf576, buf566, buf563, buf561, buf558, buf554, buf546, buf537, buf527, buf524, buf522, buf519, buf515, buf507, buf498, buf488, buf485, buf483, buf480, buf476, buf469, buf460, buf450, buf447, buf445, buf442, buf438, buf431, buf422, buf412, buf409, buf407, buf404, buf400, buf392, buf383, buf373, buf370, buf368, buf365, buf361, buf353, buf344, buf334, buf331, buf329, buf326, buf322, buf315, buf306, buf296, buf293, buf291, buf288, buf284, buf277, buf268, buf258, buf255, buf253, buf250, buf246, buf238, buf229, buf219, buf216, buf214, buf211, buf208, buf199, buf190, buf180, buf177, buf175, buf172, buf169, buf161, buf152, buf142, buf139, buf137, buf134, buf131, buf124, buf115, buf105, buf102, buf100, buf97, buf94, buf87, buf78, buf68, buf65, buf63, buf60, buf57, buf49, buf40, buf30, buf27, buf25, buf22, buf19, buf11, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((8, 3, 192, 192), (110592, 1, 576, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_9 = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    mul_17 = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 16, 96, 96), (147456, 1, 1536, 16), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_14 = rand_strided((8, 16, 96, 96), (147456, 1, 1536, 16), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 96, 96, 96), (884736, 1, 9216, 96), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_32 = rand_strided((8, 96, 96, 96), (884736, 1, 9216, 96), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 96, 48, 48), (221184, 1, 4608, 96), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_24 = rand_strided((8, 96, 48, 48), (221184, 1, 4608, 96), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 96, 1, 1), (96, 1, 96, 96), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 4, 1, 1), (4, 1, 4, 4), device='cuda:0', dtype=torch.float32)
    mul_41 = rand_strided((8, 4, 1, 1), (4, 1, 4, 4), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 96, 1, 1), (96, 1, 96, 96), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((8, 96, 48, 48), (221184, 1, 4608, 96), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_29 = rand_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_57 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_39 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cuda:0', dtype=torch.float32)
    mul_66 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
    mul_67 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_45 = rand_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_82 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 144, 24, 24), (82944, 1, 3456, 144), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_55 = rand_strided((8, 144, 24, 24), (82944, 1, 3456, 144), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
    mul_92 = rand_strided((8, 144, 24, 24), (82944, 1, 3456, 144), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_60 = rand_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_107 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_70 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cuda:0', dtype=torch.float32)
    mul_116 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    mul_117 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_76 = rand_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_132 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 240, 12, 12), (34560, 1, 2880, 240), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_86 = rand_strided((8, 240, 12, 12), (34560, 1, 2880, 240), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cuda:0', dtype=torch.float32)
    mul_141 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    mul_142 = rand_strided((8, 240, 12, 12), (34560, 1, 2880, 240), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_91 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_157 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_101 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    mul_166 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    mul_167 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_107 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_182 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_117 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    mul_191 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    mul_192 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_123 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_207 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_133 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    mul_216 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    mul_217 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_139 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_232 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_149 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    mean_9 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    mul_241 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    mul_242 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_154 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_257 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_164 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    mean_10 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    mul_266 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_267 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_170 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_282 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_180 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    mean_11 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    mul_291 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_292 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_186 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_307 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_196 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    mean_12 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    mul_316 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_317 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_202 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_332 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 672, 6, 6), (24192, 1, 4032, 672), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_212 = rand_strided((8, 672, 6, 6), (24192, 1, 4032, 672), device='cuda:0', dtype=torch.float32)
    mean_13 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    mul_341 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_342 = rand_strided((8, 672, 6, 6), (24192, 1, 4032, 672), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_217 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_357 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_227 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    mean_14 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    mul_366 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    mul_367 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_233 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_382 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_243 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    mean_15 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    mul_391 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    mul_392 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_249 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_407 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    convolution_81 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_259 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    mean_16 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    mul_416 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    mul_417 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    convolution_84 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_265 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda:0', dtype=torch.float32)
    convolution_85 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_432 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    convolution_86 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_275 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    mean_17 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    convolution_87 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    mul_441 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    mul_442 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    convolution_89 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_281 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda:0', dtype=torch.float32)
    convolution_90 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_457 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    convolution_91 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_291 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    mean_18 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    convolution_92 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    mul_466 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    convolution_93 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    mul_467 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    convolution_94 = rand_strided((8, 320, 6, 6), (11520, 1, 1920, 320), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_296 = rand_strided((8, 320, 6, 6), (11520, 1, 1920, 320), device='cuda:0', dtype=torch.float32)
    convolution_95 = rand_strided((8, 1280, 6, 6), (46080, 1, 7680, 1280), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_484 = rand_strided((8, 1280, 6, 6), (46080, 1, 7680, 1280), device='cuda:0', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_524 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_564 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_604 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_644 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_684 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda:0', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_724 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_764 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_804 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_844 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda:0', dtype=torch.float32)
    unsqueeze_558 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_884 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_606 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_924 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_964 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    unsqueeze_666 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_678 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1004 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda:0', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1044 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda:0', dtype=torch.float32)
    unsqueeze_738 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_750 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1084 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda:0', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_786 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_798 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1124 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda:0', dtype=torch.float32)
    unsqueeze_810 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_822 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1164 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda:0', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_858 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_870 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1204 = rand_strided((8, 96, 96, 96), (884736, 1, 9216, 96), device='cuda:0', dtype=torch.float32)
    unsqueeze_882 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_894 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_906 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1244 = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda:0', dtype=torch.float32)
    unsqueeze_918 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_118, primals_119, primals_121, primals_123, primals_124, primals_125, primals_126, primals_128, primals_130, primals_131, primals_132, primals_133, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_144, primals_145, primals_146, primals_147, primals_149, primals_151, primals_152, primals_153, primals_154, primals_156, primals_158, primals_159, primals_160, primals_161, primals_163, primals_165, primals_166, primals_167, primals_168, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_187, primals_188, primals_189, primals_191, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_233, primals_235, primals_236, primals_237, primals_238, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_249, primals_250, primals_427, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, add_9, mean, convolution_2, mul_16, convolution_3, mul_17, convolution_4, squeeze_7, add_14, convolution_5, squeeze_10, mul_32, convolution_6, squeeze_13, add_24, mean_1, convolution_7, mul_41, convolution_8, mul_42, convolution_9, squeeze_16, add_29, convolution_10, squeeze_19, mul_57, convolution_11, squeeze_22, add_39, mean_2, convolution_12, mul_66, convolution_13, mul_67, convolution_14, squeeze_25, add_45, convolution_15, squeeze_28, mul_82, convolution_16, squeeze_31, add_55, mean_3, convolution_17, mul_91, convolution_18, mul_92, convolution_19, squeeze_34, add_60, convolution_20, squeeze_37, mul_107, convolution_21, squeeze_40, add_70, mean_4, convolution_22, mul_116, convolution_23, mul_117, convolution_24, squeeze_43, add_76, convolution_25, squeeze_46, mul_132, convolution_26, squeeze_49, add_86, mean_5, convolution_27, mul_141, convolution_28, mul_142, convolution_29, squeeze_52, add_91, convolution_30, squeeze_55, mul_157, convolution_31, squeeze_58, add_101, mean_6, convolution_32, mul_166, convolution_33, mul_167, convolution_34, squeeze_61, add_107, convolution_35, squeeze_64, mul_182, convolution_36, squeeze_67, add_117, mean_7, convolution_37, mul_191, convolution_38, mul_192, convolution_39, squeeze_70, add_123, convolution_40, squeeze_73, mul_207, convolution_41, squeeze_76, add_133, mean_8, convolution_42, mul_216, convolution_43, mul_217, convolution_44, squeeze_79, add_139, convolution_45, squeeze_82, mul_232, convolution_46, squeeze_85, add_149, mean_9, convolution_47, mul_241, convolution_48, mul_242, convolution_49, squeeze_88, add_154, convolution_50, squeeze_91, mul_257, convolution_51, squeeze_94, add_164, mean_10, convolution_52, mul_266, convolution_53, mul_267, convolution_54, squeeze_97, add_170, convolution_55, squeeze_100, mul_282, convolution_56, squeeze_103, add_180, mean_11, convolution_57, mul_291, convolution_58, mul_292, convolution_59, squeeze_106, add_186, convolution_60, squeeze_109, mul_307, convolution_61, squeeze_112, add_196, mean_12, convolution_62, mul_316, convolution_63, mul_317, convolution_64, squeeze_115, add_202, convolution_65, squeeze_118, mul_332, convolution_66, squeeze_121, add_212, mean_13, convolution_67, mul_341, convolution_68, mul_342, convolution_69, squeeze_124, add_217, convolution_70, squeeze_127, mul_357, convolution_71, squeeze_130, add_227, mean_14, convolution_72, mul_366, convolution_73, mul_367, convolution_74, squeeze_133, add_233, convolution_75, squeeze_136, mul_382, convolution_76, squeeze_139, add_243, mean_15, convolution_77, mul_391, convolution_78, mul_392, convolution_79, squeeze_142, add_249, convolution_80, squeeze_145, mul_407, convolution_81, squeeze_148, add_259, mean_16, convolution_82, mul_416, convolution_83, mul_417, convolution_84, squeeze_151, add_265, convolution_85, squeeze_154, mul_432, convolution_86, squeeze_157, add_275, mean_17, convolution_87, mul_441, convolution_88, mul_442, convolution_89, squeeze_160, add_281, convolution_90, squeeze_163, mul_457, convolution_91, squeeze_166, add_291, mean_18, convolution_92, mul_466, convolution_93, mul_467, convolution_94, squeeze_169, add_296, convolution_95, squeeze_172, view, permute_1, mul_484, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_524, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_564, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_604, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_644, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_684, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_724, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_764, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_804, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_844, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_884, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_924, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_964, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_1004, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1044, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1084, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1124, unsqueeze_810, unsqueeze_822, unsqueeze_834, mul_1164, unsqueeze_846, unsqueeze_858, unsqueeze_870, mul_1204, unsqueeze_882, unsqueeze_894, unsqueeze_906, mul_1244, unsqueeze_918, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tinynet_a', benchmark_compiled_module)
