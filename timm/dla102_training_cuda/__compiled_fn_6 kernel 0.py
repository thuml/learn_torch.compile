
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


# kernel path: /tmp/torchinductor_youkaichao/rh/crhdqbmfhqxh33jagwlqfsbltebc34zla7cz4vaq6ius7p2gw257.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_0', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4i/c4iixjjnwylmewcbqzkiotx57vkk5uqxmpztj6sskolonhjet4wm.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*(r2 // 49)) + (2048*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 49.0
        tmp3 = tmp1 / tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5jkedqodwpgq54odkgul4entpvme3sgxj4bukbwvlxkr3pwadj.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/az/cazr56fhp4qe5k2ykevpurb6nf46zpiqxhzqnzen7atzal4uvena.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmisoomgqpgpisd62niyhond7cmtmer2mse3yqv5v2vwriqkig7b.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1024
    x2 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.002551020408163265
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qm/cqmog2mdhdz52nje36oqvmvpftunpbfkbhpdsqehckijsh3qtafs.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp2 = tl.load(in_ptr2 + (x0 + (1024*(r2 // 49)) + (2048*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + ((49*x0) + (125440*(r2 // 49)) + (250880*x1) + (r2 % 49)), rmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr4 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = 49.0
        tmp4 = tmp2 / tmp3
        tmp5 = 0.0
        tmp6 = tl.where(tmp1, tmp5, tmp4)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp0, tmp5, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp9 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, None)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5qwvrx2ognwiapissf7fqmr6ta64pyu356wf7rr7q2k5b6pdu5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (x2 + (1024*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (49*x2) + (125440*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = 49.0
    tmp4 = tmp2 / tmp3
    tmp5 = 0.0
    tmp6 = tl.where(tmp1, tmp5, tmp4)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp0, tmp5, tmp8)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.002551020408163265
    tmp15 = tmp13 * tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = tmp9 - tmp19
    tmp22 = tmp21 * tmp14
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (1024*y3)), tmp26, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iz/ciznd7l5zpuzouucpj726jnrwbbbwgb5x7xfncu56nyobm4p46i7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((49*x0) + (25088*(r2 // 49)) + (50176*x1) + (r2 % 49)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c635lzqj7ujigzwlbkle4y64ktacsiqv2qllr4wcfot4unayryxg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3iuzhyvrugpvwh5ojtfejxogl3dpkecabg3pfpszgyiqnhamwl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/za/czagvfqt62vxfrbeeh4gu3lemwi27g22ykupmd5gcvxrpe5gpqlk.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (49*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.002551020408163265
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqt3sllcke5cc347w5cjmyjwqsodpwnk4l2copab557sipeqvsl.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]

triton_poi_fused_add_div_threshold_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*i1', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (50176*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (50176 + x2 + (49*y0) + (125440*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (1024*x2) + (50176*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (y0 + (1024*x2) + (50176*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp6 = tl.load(in_ptr4 + (y3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (x2 + (49*y0) + (125440*y1)), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp13 = tmp3 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.where(tmp2, tmp1, tmp15)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrtbedwtonfe3rf66772glwozmecyxrbx3yubljan5pffjulhsk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcg3tc46gkreyifbqqih5wml3254xmht5ukedzjbjclo2fm54xi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (50176*(r2 // 49)) + (100352*x1) + (r2 % 49)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lr3lnx3c7crjtiaqnpu6ego5keeudkqeyv5paei2xle2sk36nl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.002551020408163265
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (1024*y3)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4p/c4p4c4kuezridvgnimv5vioithsqwl2woo2reilsork54ddmdsit.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x1) + (100352*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clvpqfqwh4xhdlik5o44adwa7tgdtdcdjsbicz234dm5imvgpif7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5yppmd4ghw4psvxfq27xe57cy2uadwl65fdgy7nxph5r2ix4gj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x0) + (100352*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c463g7og5ilq4s5ccbfmwo55awllq2i27eqz7hey7cpn65675cmr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co353eesdnd4dfbblqbl74eg53ennoyensz3ohzi46jris362pgw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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
    tmp5 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdlt6bldfp54qnikyukw7tddwcyjskpjqdfgcv2l4yfdqzlr4bd.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (100352 + x0 + (125440*x1)), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cveee32ygih3jrqjk3wrvgt7ay6u2jrwuc4s7qncmo354zaixdnc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x0) + (100352*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp5, tmp4, tmp8)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.load(in_ptr3 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp9 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/go/cgo7bd6k57645pfn7deqcmleuim77av5ppystyqcgajvxupirr6o.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lg/clgaot6orxsndjwncoe72yv52vyod5xaetkxv4x3obdi422wfggj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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
    tmp4 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/co/ccocljvzzidptt7zydznt2ua44cm2o2j7wrgkh3kdhdcj2xzscyi.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (196*x2) + (551936*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp4 + tmp5
    tmp7 = tl.where(tmp3, tmp2, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp0, tmp2, tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (512*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2elmvq3wxwz4pd7qji2qwyoojy6hmacfvrqkfdr35lszrmxo4b6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
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
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp3 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/or/corpjq2oibfqfuxmmbqvgdstcxpg6qyp7lfru556jhxxnhtrd2jo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47d3zbm6zk2gfqaz24j5qpa4jypbhrcfxou7iivsbqrlaaymqjh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (256*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x1) + (50176*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mf/cmfkwypucxth2f3kaqknoxte6blc2i6wyvnrjxxwltis4qw6hz5g.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgx57522je4mzj33pzsy75lellslidzw5fmnzdlm6r7vtt73s7k.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x0) + (50176*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (256*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwg7uw423umoobjcpnrjk2sezqko223dxssz7l3gykbve5jtmygs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jswfgf3iisker7wpccbknbucdnxlbu5oyte5nirv55ujgycb5t.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2r/c2ruvswvqh3nw5i72p5vgyl3mzd5moqrw7fazzswdl53q3quyrhs.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (100352 + (196*x1) + (551936*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.load(in_ptr3 + ((196*x1) + (100352*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.where(tmp5, tmp4, tmp10)
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mf/cmfybayzjvskcx2esccrfuab3qejcckty6zqcnl6c44auhcij6va.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (100352 + (196*x0) + (551936*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.load(in_ptr3 + ((196*x0) + (100352*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.where(tmp5, tmp4, tmp10)
        tmp12 = tl.load(in_ptr4 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czlcnouvectuo5go3wktc73bfxrm42wagsmg63kpmmunfiyr2r5a.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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
    tmp3 = tl.load(in_ptr1 + (100352 + y0 + (196*x2) + (551936*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.0006377551020408163
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (512*y3)), tmp25, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gp/cgpmylcvty2vg5mnop3jw4vtnyg2we5vyqq2ttgh6t4apsldbx6d.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (451584 + x2 + (196*y0) + (551936*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (100352 + x2 + (196*y0) + (551936*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp4 <= tmp1
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp5, tmp1, tmp10)
    tmp12 = tmp3 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfp7kz564exa2yc5dxu47iy72dgkem3mx65ux3isrdjsyplisgb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2b/c2bpjgb3dmefgel2moadao75krqoo5vy463knoknyckxverfeinl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (100352*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dz/cdzwhjcamrywe4mp7twesjigieprrfmstws44ws2kjomg3kaaooe.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhoiaupssdyb23egwe3ctwknxxi4ghadtlf6zwfml2ni2ilxcgo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccp2ngpot2yuftsfwdrk5rmivchiogkuecpzdr6eg2pyjwvk24x7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ng/cnge3lpf7y7ii4im5eoihmunytkjpw3mlsn4cqlrgtepl34uc76b.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((196*x1) + (100352*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((196*x1) + (200704*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp3, tmp7, tmp6)
        tmp9 = tl.load(in_ptr3 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/el/celyrrgpak74cefx5ghujvdx5zp3pcosq4s3gioogaruu7etj656.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.0006377551020408163
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxb6qvdgq4lzlqvqbog6pmjn4gfejsnur7q4g3ps4ubd4nqgixgh.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (100352 + x2 + (196*y0) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2 + (196*y0) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp9 = tmp3 + tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp2, tmp1, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q5/cq5wef4qvczb5vr3vwnwiha5zsrarwhfzlft6rt7g555kwle2wji.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (351232 + r1 + (196*x0) + (551936*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7f/c7f5o56iofbgkmjjsqf4dxbeiowixgvtemn3o3k55aabzbukogrw.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (351232 + x2 + (196*y0) + (551936*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.0006377551020408163
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr1 + (y0 + (512*x2) + (100352*y1)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3n/c3nnqgck5zaoav7lq3oeiykwijkrgofxdeiuq2jzvixmxce2rt44.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (351232 + x2 + (196*y0) + (551936*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2 + (196*y0) + (301056*y1)), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp3, tmp2, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp0, tmp2, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgp4bobo5nven6d7yjoxomlzzm42bn7e6ygordqkcx6lzyzch72.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (100352 + r1 + (196*x0) + (301056*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgwmpij66pl4cs6mz2bwgg7drchzi5373hw35rlgls4rmffhfxu.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (100352 + x2 + (196*y0) + (301056*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.0006377551020408163
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr1 + (y0 + (512*x2) + (100352*y1)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cnc56mv5fogosmsq2ckndiemi26k6j647djqyl6wlsqy6g2l2fwe.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (200704 + x2 + (196*y0) + (301056*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (100352 + x2 + (196*y0) + (301056*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp4 <= tmp1
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp5, tmp1, tmp10)
    tmp12 = tmp3 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuzxiw5uoscri7lojh627fzcw6v372k7v4rcpwcqxoqetsqbwpf5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (250880 + r1 + (196*x0) + (551936*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwmnz3sdgioira2wci4z7fp2xgcj4z2pag73aeyi7buktvjn7bki.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (250880 + x2 + (196*y0) + (551936*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.0006377551020408163
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr1 + (y0 + (512*x2) + (100352*y1)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xc/cxc2lool7pmlwemxtwqa7u2avs4yiwgl756u65w2knbycev5roxw.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_52', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (250880 + x2 + (196*y0) + (551936*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2 + (196*y0) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp3, tmp2, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp0, tmp2, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rg/crg4e6yzft5hc5uldqsfi5exgit5ybjgdufxhf7k72kjl2fdoojx.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (100352 + r1 + (196*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqwidtkblfyqqyheh5ndlv5ublugfsa6t5dpb4rzcw2mhcmpi6o.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (100352 + x2 + (196*y0) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.0006377551020408163
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr1 + (y0 + (512*x2) + (100352*y1)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crntvjbmzjboowewmehqts7qxu6lvwhk7buv5avz6bgznw2d22un.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_55', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (301056 + x2 + (196*y0) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (100352 + x2 + (196*y0) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp4 <= tmp1
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp5, tmp1, tmp10)
    tmp12 = tmp3 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/cki4ekmyze5q3267ixathj3ljnhwp7g2ctxtodut2hhlysg7ydh4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + r1 + (196*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5iaefom46lnq2o5pag46bccrirziaeb2utdrvq22o5e4f4m6oo.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (200704 + x2 + (196*y0) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.0006377551020408163
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr1 + (y0 + (512*x2) + (100352*y1)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3v3fceat2ha2skdu4u4p3nwhrsq4cv5wiqm3pqbfen4q7qfbxhm.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_58', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (200704 + x2 + (196*y0) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2 + (196*y0) + (301056*y1)), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp3, tmp2, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp0, tmp2, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ch/cchdbipzgt2ahcrgflb3uptpjdl6isnz5kjfekp57mldne2esgrl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (100352*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr3 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp3 * tmp15
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5jmzletjdddbflzz22euiir3wh3kxzntn75rinnhauehybdrca.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (512*y3)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6tegpgrbkjkanzhyyc3l5xsd4vlddbko5flxqqosixkf67bwfu7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (200704*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74la6czrbrstuuudehoez5hmmj4vf5o3jjbuyrqzdoxsy7os3bt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6z/c6zmuic37y4dubt6amyvggmfrjsgrn62wk5ankaahk3g3yspdouz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x0) + (200704*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6u/c6u5nvflldrj32v2xwk7wc3navfxqyu6maqoyadk65nu4inmat5g.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2cvglw7xzyut2ur3otnoizzybdig3ana3lhk3ctnyq3q5ykesm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmwihumie7eai3dxvafz7yuugkvtscqjk2emyox7wd2zbmz2gdr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x0) + (200704*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4r/c4rmjw5bsp2fpbjyzzpg5j4xniy5qfmd3i5hpssmil3eks7bvotg.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2tlhourgrvnrpedqqbi2bljyopx5seyrympeldrz4z7gptvuxo.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_68', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.00015943877551020407
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp25, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2hvq2zyatdcy4vf2q222ssfar2yfq6uqgi4bbrwzuovngsjgi6.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_69', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0 + (784*x2) + (903168*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp3, tmp2, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp0, tmp2, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/caw3d3s2pqkol4i2rokbepgfsomt6gjpze4sq7m4fiexvtc4z4bj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/67/c676mpfi7bwjxk2k4f6fpl3fsw563qtzveenkbzmqleebetcjtrf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00015943877551020407
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uy/cuybd2vqktklp4suzpucfox3bbwjuoabepcwgxhf6urjfhuwplky.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (100352*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmz3sypkumchqnxd5adcydrysnn3ywgt733kkyqrnltk3jidhnrj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf25j2qlvoqqfqcen67maofawfjumq4r7mehsfortwtjrile3ucs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x0) + (100352*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhkizmgsj34wome6mek6txzug52x7vsonpayd3fd2jts5rydtso.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xi/cxikasqk3xnzpvhvfoa65pdvus5rra4olsatqnmrlzlawhnbvmn2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oo/cooqftpcfihwtmo4csjmae3q6lipdqzdzcdgsqoct2ov6zirmnbs.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + (784*x1) + (903168*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((784*x1) + (200704*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mt/cmtlkr7di25yjbar3uov6bkx4duymo2iv4vq57twigm2wabhfzly.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp10 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + (784*x0) + (903168*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((784*x0) + (200704*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr4 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbzjyh5jvpnvgrphv3ncqt2lsyemgebixmjl472ot7s5gmlxnmv.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_79', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (200704 + y0 + (784*x2) + (903168*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.00015943877551020407
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp25, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cot3igktsu4r4ltzogubnj4qfp4nzj65ucsowljnftn2ljezvkpo.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_80', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (702464 + x2 + (784*y0) + (903168*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (200704 + x2 + (784*y0) + (903168*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp4 <= tmp1
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp5, tmp1, tmp10)
    tmp12 = tmp3 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvgu5lntszfbshcxxzi6dziu5kpy4w4yjm4jvtghoxayopu2g5z.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdb7b6lzjmwlnovt7wyhwsbfxvpkh2xrmtrztt7qucgtmvb6wnjb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (200704*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s6/cs67ua7we7aqrn3zckr2zkcmnik5fhwn2r6lwy65rqjrhxmc2gfw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bh/cbh3sl4tlhttg3izt4igiejofgz6kasbtxc6xid7tgxlkgiumptg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00015943877551020407
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5u/c5ubzqzmyolfzhao3r2hyrxud5dyjzlajzocmlhmqswbxe5x4klp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co6wawscwubmct2apuun4yfkbqgxi5hxodnqo3wbl5be5qwlow67.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((784*x1) + (200704*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((784*x1) + (401408*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crpbs4so7bcsgs3q27gihtwrugn32t3i7tjqldmty5hd2rg7h5vz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (784*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00015943877551020407
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n5/cn5kstmwqe3vixpffpkguri6beeehvua5pldsfzirmw3u7qzicnz.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_88', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (200704 + x2 + (784*y0) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2 + (784*y0) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp9 = tmp3 + tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp2, tmp1, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwykrg5b2qhydvcuwk2o46tcsl7ynfzneu5bkalcbuhuqbo74yan.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (501760 + r1 + (784*x0) + (903168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fy/cfym3n7y7fzrpdekgvxlqgkl3rl57wvvgsyssygxyxq3u6iqefqp.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (501760 + x2 + (784*y0) + (903168*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.00015943877551020407
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr1 + (y0 + (256*x2) + (200704*y1)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwgzbd7qbhep3hcfhlcznkglhoeobw3ddjmadtyndp47ixyj6rty.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_91', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (501760 + x2 + (784*y0) + (903168*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2 + (784*y0) + (602112*y1)), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp3, tmp2, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp0, tmp2, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/czt5pnyy424y6l3fendqddw4nhx2k3cbeueuhvyhegahxrbdmd6d.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + r1 + (784*x0) + (602112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ap/capoo2qja5ifjunozeyxthb4wfnopc5c37bxgarup3gp64avqn6r.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (200704 + x2 + (784*y0) + (602112*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.00015943877551020407
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr1 + (y0 + (256*x2) + (200704*y1)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rs/crsfsjwrgkvokex2p55co7mgyrx667ja6p6eckys7h5xkbq7or5f.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_94', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (401408 + x2 + (784*y0) + (602112*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (200704 + x2 + (784*y0) + (602112*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp4 <= tmp1
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp5, tmp1, tmp10)
    tmp12 = tmp3 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j4/cj423nzeyvze4p776yncarsa6fnlsgzwcsnjem2jpyjz56b2g7rm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (200704*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfskrttm66ri5b5ghz7czmjwkr44smcpm2e4mha3oxlj3roty6w5.py
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
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00015943877551020407
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (256*y3)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2t/c2ta7lfnoqqmreshec7j5eq2omwht7b5nrm2ul72j6y62k6kdfla.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (401408*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqkb4asfdz2sahn6jb7by6pn3na5bdqlo3wt5sbqf2r2syh3j4z.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwbaq5bo6ptmu3f42udxlyuwmoeje2dw2djq4s65jq5pu2tw4qcm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (401408*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sjrdt2gju63xny4aqkpymkxdbbmuuejpc6g7r45msfhje6icl6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 196
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
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/cru2v3cinsofth4fy5bcz2zx22zgcgyw4kgfy26vi3jr3cv3tgya.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_101', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.985969387755102e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhd2bzxxzp6w73gbq2itzelk4d4kugmskqfsc2kdpmnz723newt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (401408*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ce/cceodk3o5q3h6ms5ckqziatlwnojyuijipgxezwalyrbe37jre3j.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 196
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


# kernel path: /tmp/torchinductor_youkaichao/xu/cxupe6irso4vs2mbreo6e5thys3j6qan4okjaqg34nqdqjqjyzgc.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_104', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 3.985969387755102e-05
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp25, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2a/c2avvcyaxny4z3v3yd42mzpig7mllryaqqrm57fkz5sjfl7kadbf.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_105', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp3, tmp2, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp0, tmp2, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ya/cyawdnahzxd6qdsldo2l3xyvhivjrcdfxstcgsao5me3jouuharu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/cru6j22ccanldtau44o2egisjjnyez42wdxw3afzbjymbcs6fgc7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 3.985969387755102e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y4/cy44vsg36vyvozmpkf7mkxit3optizc6s7cuxx663u24cgm2y6iu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (200704*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/ckly6hb2m4jnzkpbeym5ekf66ic2lrnh6swrjuoq576jy7hhhlq5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_109 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_109', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyzdes57doxdaihr2j6x36f6rmczuzzidl2fb3he3jttyi4ptt7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_110', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (200704*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmt52nrl2g3zeitkrs3rmzbu6fgg3crm43mlwhhrmheso3aqmnb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_111 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 196
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
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dq/cdqnxupqhtsmhh72n7apksfndqku6aml3auriyj3s46uzhgm2iq4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_112 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_112', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.985969387755102e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfca2etyewrj55wn3y4tsjfq4gtzsbgwb3cuko3xjid2i2lq3kj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_113', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (401408 + (3136*x1) + (802816*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((3136*x1) + (401408*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ng/cngnrbxovxvlos5ktqj4jscp6zvfkgjot7dtpljrp6452kjyhl7l.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_114 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp10 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp17 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (401408 + (3136*x0) + (802816*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((3136*x0) + (401408*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr4 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr6 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp8 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkvnn7lua5zjcaymp4c2yvlftbs2u47k4glqpzgbuay7dxutw5m.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_115 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_115', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (401408 + y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr14 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 3.985969387755102e-05
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp25 = tmp23 - tmp24
    tmp27 = tmp26 * tmp13
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp25 * tmp30
    tmp32 = tmp8 - tmp31
    tmp33 = tmp32 - tmp21
    tmp35 = tmp15 * tmp34
    tmp36 = tmp22 * tmp35
    tmp38 = tmp28 * tmp37
    tmp39 = tmp33 * tmp38
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp36, xmask & ymask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x2 + (128*y3)), tmp39, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kbe7i7sj6xebmewntic54c5qoxsorry2etlvgka7wfjhykad4p.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x1) + (802816*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/57/c57xofpmsfr6mtz34q7bbuf2v5arcyg5ikqi4ptwta2y36ueacnn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_117 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_117', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwm5h5kp5uvxv7aelxq6m4dhoxtftyyduc4233trnuokcrdvf77w.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_118', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x0) + (802816*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gz/cgzbpoqy32ewe2mt4574lpa7kr3zfdickt6smupzqgzg4l2uuicp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_119 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 784
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
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvz2cb2tbds7qm6q7qk5fjewj3cuvkah5qy2myu7xgd56lkovj3t.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_120 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_120', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 9.964923469387754e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrvvpnltevcautwbygxr72qopmi3nqhhggf5qfdfab4e7ztrene.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_121', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x0) + (401408*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ty/ctygd6snypl2pnp3cvcgnmw22aqz5j4owbbw6246tekewntsp2hh.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_122 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_122', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 784
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
''')


# kernel path: /tmp/torchinductor_youkaichao/5z/c5zos2lipwnfsfgbxsjouwms42uoa5f65zzb6vhgi3gynzbbmxnw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_123 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_123', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 784
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


# kernel path: /tmp/torchinductor_youkaichao/fa/cfapxhfgo7td4rjvyoazwwxtgov4wod3kebjhcnouerrh2peszvr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_124 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_124', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 9.964923469387754e-06
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
    tl.store(in_out_ptr0 + (x2 + (32*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnp5fkh2mpnjmenel54cpbg47bgjupfydqt4pfajrgvudmhse2km.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_125', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((50176*x1) + (802816*((r2 + (128*x0)) // 50176)) + ((r2 + (128*x0)) % 50176)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ss/cssrfxgy4scq3iodbbncxylbyaubk7iuonqzbl5wu4k7ypyj637q.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_126 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_126', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 3136
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
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cw/ccwrcstnx5546p73vyyhsvqdx7fg3eg2qmplaxugepj3w2i3kd2l.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((50176*x0) + (802816*((r2 + (128*x1)) // 50176)) + ((r2 + (128*x1)) % 50176)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkepirdpglxgfurtat3eivrdfbt26i5kqqi77arq2vdbm5zz336.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_128 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16, 4096],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_128', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 3136
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
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wu/cwuttfvdpsutxfb2j2adjqikuigzuh3zvnhiit3seusrjgdk46hb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_129 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 401408
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (50176*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 2.4912308673469386e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp21, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, primals_314, primals_316, primals_633, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, cat, convolution_10, squeeze_31, relu_9, getitem_24, getitem_25, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_10, convolution_13, squeeze_40, relu_11, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, cat_1, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, cat_2, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, relu_24, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, relu_26, convolution_29, squeeze_88, relu_27, convolution_30, squeeze_91, relu_28, convolution_31, squeeze_94, cat_3, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, relu_31, convolution_34, squeeze_103, relu_32, convolution_35, squeeze_106, relu_33, convolution_36, squeeze_109, relu_34, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, cat_4, convolution_39, squeeze_118, relu_37, getitem_88, getitem_89, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, relu_42, convolution_46, squeeze_139, cat_5, convolution_47, squeeze_142, relu_44, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, relu_46, convolution_50, squeeze_151, relu_47, convolution_51, squeeze_154, relu_48, convolution_52, squeeze_157, relu_49, convolution_53, squeeze_160, cat_6, convolution_54, squeeze_163, relu_51, convolution_55, squeeze_166, relu_52, convolution_56, squeeze_169, relu_53, convolution_57, squeeze_172, relu_54, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, relu_56, convolution_60, squeeze_181, cat_7, convolution_61, squeeze_184, relu_58, convolution_62, squeeze_187, relu_59, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, relu_61, convolution_65, squeeze_196, relu_62, convolution_66, squeeze_199, relu_63, convolution_67, squeeze_202, cat_8, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, relu_66, convolution_70, squeeze_211, relu_67, convolution_71, squeeze_214, relu_68, convolution_72, squeeze_217, relu_69, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, cat_9, convolution_75, squeeze_226, relu_72, convolution_76, squeeze_229, relu_73, convolution_77, squeeze_232, relu_74, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, relu_76, convolution_80, squeeze_241, relu_77, convolution_81, squeeze_244, cat_10, convolution_82, squeeze_247, relu_79, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, relu_81, convolution_85, squeeze_256, relu_82, convolution_86, squeeze_259, relu_83, convolution_87, squeeze_262, relu_84, convolution_88, squeeze_265, cat_11, convolution_89, squeeze_268, relu_86, convolution_90, squeeze_271, relu_87, convolution_91, squeeze_274, relu_88, convolution_92, squeeze_277, relu_89, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, relu_91, convolution_95, squeeze_286, cat_12, convolution_96, squeeze_289, relu_93, getitem_210, getitem_211, convolution_97, squeeze_292, convolution_98, squeeze_295, relu_94, convolution_99, squeeze_298, relu_95, convolution_100, squeeze_301, relu_96, convolution_101, squeeze_304, relu_97, convolution_102, squeeze_307, relu_98, convolution_103, squeeze_310, cat_13, convolution_104, squeeze_313, clone, le, unsqueeze_422, le_1, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, le_8, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, le_15, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, le_22, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, le_29, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, le_36, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, le_43, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, unsqueeze_1022, le_50, unsqueeze_1034, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, unsqueeze_1094, unsqueeze_1106, le_57, unsqueeze_1118, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, le_64, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, le_71, unsqueeze_1298, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, le_78, unsqueeze_1382, unsqueeze_1394, unsqueeze_1406, unsqueeze_1418, unsqueeze_1430, unsqueeze_1442, unsqueeze_1454, le_85, unsqueeze_1466, unsqueeze_1478, unsqueeze_1490, unsqueeze_1502, unsqueeze_1514, unsqueeze_1526, unsqueeze_1538, unsqueeze_1550, le_92, unsqueeze_1562, unsqueeze_1574, unsqueeze_1586, unsqueeze_1598, unsqueeze_1610, unsqueeze_1622, unsqueeze_1634, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_4, (16, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (32, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_10, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_16, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_22, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_25, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_28, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_34, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_40, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_43, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_46, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_52, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_55, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_58, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_61, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_64, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_67, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_70, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_73, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_76, (256, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_79, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_82, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_85, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_88, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_91, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_92, (128, ), (1, ))
    assert_size_stride(primals_94, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_97, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_100, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_103, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_106, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_109, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_112, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_115, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_118, (256, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_121, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_124, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_127, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_130, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_133, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_136, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_139, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_140, (512, ), (1, ))
    assert_size_stride(primals_142, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_145, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_148, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_151, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_152, (512, ), (1, ))
    assert_size_stride(primals_154, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_155, (256, ), (1, ))
    assert_size_stride(primals_157, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_160, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_161, (512, ), (1, ))
    assert_size_stride(primals_163, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_166, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_167, (256, ), (1, ))
    assert_size_stride(primals_169, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_170, (256, ), (1, ))
    assert_size_stride(primals_172, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_175, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_176, (256, ), (1, ))
    assert_size_stride(primals_178, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_179, (256, ), (1, ))
    assert_size_stride(primals_181, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_182, (512, ), (1, ))
    assert_size_stride(primals_184, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_185, (512, ), (1, ))
    assert_size_stride(primals_187, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_188, (256, ), (1, ))
    assert_size_stride(primals_190, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_191, (256, ), (1, ))
    assert_size_stride(primals_193, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_196, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_197, (256, ), (1, ))
    assert_size_stride(primals_199, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_202, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_205, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_206, (512, ), (1, ))
    assert_size_stride(primals_208, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_209, (256, ), (1, ))
    assert_size_stride(primals_211, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_214, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_217, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_220, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_221, (256, ), (1, ))
    assert_size_stride(primals_223, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_224, (512, ), (1, ))
    assert_size_stride(primals_226, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_227, (512, ), (1, ))
    assert_size_stride(primals_229, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_232, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_235, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_236, (512, ), (1, ))
    assert_size_stride(primals_238, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_239, (256, ), (1, ))
    assert_size_stride(primals_241, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_244, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_245, (512, ), (1, ))
    assert_size_stride(primals_247, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_248, (512, ), (1, ))
    assert_size_stride(primals_250, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_253, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_254, (256, ), (1, ))
    assert_size_stride(primals_256, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_257, (512, ), (1, ))
    assert_size_stride(primals_259, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_262, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_265, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_266, (512, ), (1, ))
    assert_size_stride(primals_268, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_269, (512, ), (1, ))
    assert_size_stride(primals_271, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_272, (256, ), (1, ))
    assert_size_stride(primals_274, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_275, (256, ), (1, ))
    assert_size_stride(primals_277, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_278, (512, ), (1, ))
    assert_size_stride(primals_280, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_281, (256, ), (1, ))
    assert_size_stride(primals_283, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_284, (256, ), (1, ))
    assert_size_stride(primals_286, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_287, (512, ), (1, ))
    assert_size_stride(primals_289, (512, 2816, 1, 1), (2816, 1, 1, 1))
    assert_size_stride(primals_290, (512, ), (1, ))
    assert_size_stride(primals_292, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_293, (1024, ), (1, ))
    assert_size_stride(primals_295, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_296, (512, ), (1, ))
    assert_size_stride(primals_298, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_299, (512, ), (1, ))
    assert_size_stride(primals_301, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_304, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_307, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_308, (512, ), (1, ))
    assert_size_stride(primals_310, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_313, (1024, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(primals_314, (1024, ), (1, ))
    assert_size_stride(primals_316, (1000, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_633, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 16, 224, 224), (802816, 1, 3584, 16))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(relu, (8, 16, 224, 224), (802816, 1, 3584, 16))
    assert_size_stride(convolution_1, (8, 16, 224, 224), (802816, 1, 3584, 16))
    assert_size_stride(squeeze_4, (16, ), (1, ))
    assert_size_stride(relu_1, (8, 16, 224, 224), (802816, 1, 3584, 16))
    assert_size_stride(convolution_2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_7, (32, ), (1, ))
    assert_size_stride(relu_2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(getitem_6, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(getitem_7, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_3, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_10, (128, ), (1, ))
    assert_size_stride(convolution_4, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(relu_3, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_5, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(relu_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_6, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_19, (128, ), (1, ))
    assert_size_stride(relu_5, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_7, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(relu_6, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_8, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_7, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_9, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_28, (128, ), (1, ))
    assert_size_stride(cat, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_10, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_31, (128, ), (1, ))
    assert_size_stride(relu_9, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(getitem_24, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(getitem_25, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_11, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_34, (256, ), (1, ))
    assert_size_stride(convolution_12, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_37, (128, ), (1, ))
    assert_size_stride(relu_10, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_13, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_40, (128, ), (1, ))
    assert_size_stride(relu_11, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_14, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_43, (256, ), (1, ))
    assert_size_stride(relu_12, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_15, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(relu_13, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_16, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_49, (128, ), (1, ))
    assert_size_stride(relu_14, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_17, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_52, (256, ), (1, ))
    assert_size_stride(cat_1, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_18, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_55, (256, ), (1, ))
    assert_size_stride(relu_16, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_19, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(relu_17, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_20, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_61, (128, ), (1, ))
    assert_size_stride(relu_18, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_21, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_64, (256, ), (1, ))
    assert_size_stride(relu_19, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_22, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(relu_20, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_23, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_70, (128, ), (1, ))
    assert_size_stride(relu_21, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_24, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(cat_2, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_25, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_76, (256, ), (1, ))
    assert_size_stride(relu_23, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_26, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_79, (128, ), (1, ))
    assert_size_stride(relu_24, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_27, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_82, (128, ), (1, ))
    assert_size_stride(relu_25, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_28, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(relu_26, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_29, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_88, (128, ), (1, ))
    assert_size_stride(relu_27, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_30, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_91, (128, ), (1, ))
    assert_size_stride(relu_28, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_31, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_94, (256, ), (1, ))
    assert_size_stride(cat_3, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_32, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_97, (256, ), (1, ))
    assert_size_stride(relu_30, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_33, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_100, (128, ), (1, ))
    assert_size_stride(relu_31, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_34, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_103, (128, ), (1, ))
    assert_size_stride(relu_32, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_35, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_106, (256, ), (1, ))
    assert_size_stride(relu_33, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_36, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_109, (128, ), (1, ))
    assert_size_stride(relu_34, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_37, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_112, (128, ), (1, ))
    assert_size_stride(relu_35, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_38, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_115, (256, ), (1, ))
    assert_size_stride(cat_4, (8, 1152, 28, 28), (903168, 1, 32256, 1152))
    assert_size_stride(convolution_39, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_118, (256, ), (1, ))
    assert_size_stride(relu_37, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(getitem_88, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(getitem_89, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_40, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_121, (512, ), (1, ))
    assert_size_stride(convolution_41, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_124, (256, ), (1, ))
    assert_size_stride(relu_38, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_42, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_127, (256, ), (1, ))
    assert_size_stride(relu_39, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_43, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_130, (512, ), (1, ))
    assert_size_stride(relu_40, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_44, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_133, (256, ), (1, ))
    assert_size_stride(relu_41, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_45, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_136, (256, ), (1, ))
    assert_size_stride(relu_42, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_46, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_139, (512, ), (1, ))
    assert_size_stride(cat_5, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_47, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_142, (512, ), (1, ))
    assert_size_stride(relu_44, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_48, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_145, (256, ), (1, ))
    assert_size_stride(relu_45, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_49, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_148, (256, ), (1, ))
    assert_size_stride(relu_46, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_50, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_151, (512, ), (1, ))
    assert_size_stride(relu_47, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_51, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_154, (256, ), (1, ))
    assert_size_stride(relu_48, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_52, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_157, (256, ), (1, ))
    assert_size_stride(relu_49, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_53, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_160, (512, ), (1, ))
    assert_size_stride(cat_6, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_54, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_163, (512, ), (1, ))
    assert_size_stride(relu_51, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_55, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_166, (256, ), (1, ))
    assert_size_stride(relu_52, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_56, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_169, (256, ), (1, ))
    assert_size_stride(relu_53, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_57, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_172, (512, ), (1, ))
    assert_size_stride(relu_54, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_58, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_175, (256, ), (1, ))
    assert_size_stride(relu_55, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_59, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_178, (256, ), (1, ))
    assert_size_stride(relu_56, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_60, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_181, (512, ), (1, ))
    assert_size_stride(cat_7, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_61, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_184, (512, ), (1, ))
    assert_size_stride(relu_58, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_62, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_187, (256, ), (1, ))
    assert_size_stride(relu_59, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_63, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_190, (256, ), (1, ))
    assert_size_stride(relu_60, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_64, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_193, (512, ), (1, ))
    assert_size_stride(relu_61, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_65, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_196, (256, ), (1, ))
    assert_size_stride(relu_62, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_66, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_199, (256, ), (1, ))
    assert_size_stride(relu_63, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_67, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_202, (512, ), (1, ))
    assert_size_stride(cat_8, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    assert_size_stride(convolution_68, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_205, (512, ), (1, ))
    assert_size_stride(relu_65, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_69, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_208, (256, ), (1, ))
    assert_size_stride(relu_66, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_70, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_211, (256, ), (1, ))
    assert_size_stride(relu_67, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_71, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_214, (512, ), (1, ))
    assert_size_stride(relu_68, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_72, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_217, (256, ), (1, ))
    assert_size_stride(relu_69, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_73, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_220, (256, ), (1, ))
    assert_size_stride(relu_70, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_74, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_223, (512, ), (1, ))
    assert_size_stride(cat_9, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_75, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_226, (512, ), (1, ))
    assert_size_stride(relu_72, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_76, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_229, (256, ), (1, ))
    assert_size_stride(relu_73, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_77, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_232, (256, ), (1, ))
    assert_size_stride(relu_74, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_78, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_235, (512, ), (1, ))
    assert_size_stride(relu_75, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_79, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_238, (256, ), (1, ))
    assert_size_stride(relu_76, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_80, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_241, (256, ), (1, ))
    assert_size_stride(relu_77, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_81, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_244, (512, ), (1, ))
    assert_size_stride(cat_10, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_82, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_247, (512, ), (1, ))
    assert_size_stride(relu_79, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_83, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_250, (256, ), (1, ))
    assert_size_stride(relu_80, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_84, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_253, (256, ), (1, ))
    assert_size_stride(relu_81, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_85, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_256, (512, ), (1, ))
    assert_size_stride(relu_82, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_86, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_259, (256, ), (1, ))
    assert_size_stride(relu_83, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_87, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_262, (256, ), (1, ))
    assert_size_stride(relu_84, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_88, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_265, (512, ), (1, ))
    assert_size_stride(cat_11, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_89, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_268, (512, ), (1, ))
    assert_size_stride(relu_86, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_90, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_271, (256, ), (1, ))
    assert_size_stride(relu_87, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_91, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_274, (256, ), (1, ))
    assert_size_stride(relu_88, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_92, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_277, (512, ), (1, ))
    assert_size_stride(relu_89, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_93, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_280, (256, ), (1, ))
    assert_size_stride(relu_90, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_94, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_283, (256, ), (1, ))
    assert_size_stride(relu_91, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_95, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_286, (512, ), (1, ))
    assert_size_stride(cat_12, (8, 2816, 14, 14), (551936, 1, 39424, 2816))
    assert_size_stride(convolution_96, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_289, (512, ), (1, ))
    assert_size_stride(relu_93, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_210, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(getitem_211, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_97, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_292, (1024, ), (1, ))
    assert_size_stride(convolution_98, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_295, (512, ), (1, ))
    assert_size_stride(relu_94, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_99, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_298, (512, ), (1, ))
    assert_size_stride(relu_95, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_100, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_301, (1024, ), (1, ))
    assert_size_stride(relu_96, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_101, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_304, (512, ), (1, ))
    assert_size_stride(relu_97, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_102, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_307, (512, ), (1, ))
    assert_size_stride(relu_98, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_103, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_310, (1024, ), (1, ))
    assert_size_stride(cat_13, (8, 2560, 7, 7), (125440, 1, 17920, 2560))
    assert_size_stride(convolution_104, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_313, (1024, ), (1, ))
    assert_size_stride(clone, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(le, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(unsqueeze_422, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_1, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(unsqueeze_434, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_446, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_470, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_482, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_494, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_518, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_8, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_530, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_542, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_554, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_566, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_578, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_590, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_602, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_15, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_614, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_626, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_638, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_650, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_662, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_674, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_686, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_22, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_698, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_710, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_722, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_734, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_746, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_758, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_770, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_29, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_782, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_794, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_806, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_818, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_830, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_842, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_854, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_36, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_866, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_878, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_890, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_902, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_914, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_926, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_938, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_43, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_950, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_962, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_974, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_986, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_998, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1010, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1022, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_50, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_1034, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1046, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1058, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1070, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1082, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1094, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1106, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_57, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_1118, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1130, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1142, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1154, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1166, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1178, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1190, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1202, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_64, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_1214, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1226, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1238, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1250, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1262, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1274, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1286, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_71, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_1298, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1310, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1322, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1334, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1346, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1358, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1370, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_78, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_1382, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1394, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1406, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1418, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1430, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1442, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1454, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_85, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_1466, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1478, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1490, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1502, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1514, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1526, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1538, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1550, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_92, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(unsqueeze_1562, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1574, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1586, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1598, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1610, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1622, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1634, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1646, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_1658, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1670, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_convolution_backward_0.run(tangents_1, buf0, 1000, 8, grid=grid(1000), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1 = aten.convolution_backward(reinterpret_tensor(tangents_1, (8, 1000, 1, 1), (1000, 1, 1, 1), 0), clone, primals_316, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone
        del primals_316
        del tangents_1
        buf2 = buf1[0]
        buf3 = buf1[1]
        del buf1
        buf4 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf2, convolution_104, unsqueeze_422, buf4, buf6, 4096, 98, grid=grid(4096), stream=stream0)
        buf5 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf4, buf5, 1024, 4, grid=grid(1024), stream=stream0)
        buf7 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf8 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf6, squeeze_313, buf7, buf8, 1024, 4, grid=grid(1024), stream=stream0)
        buf9 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4.run(le, buf2, convolution_104, unsqueeze_422, buf7, squeeze_313, buf5, primals_314, buf9, 401408, grid=grid(401408), stream=stream0)
        del convolution_104
        del primals_314
        del squeeze_313
        del unsqueeze_422
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf10 = aten.convolution_backward(buf9, cat_13, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_13
        del primals_313
        buf11 = buf10[0]
        buf12 = buf10[1]
        del buf10
        buf13 = buf6; del buf6  # reuse
        buf15 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_5.run(le_1, le, buf2, buf11, convolution_103, unsqueeze_434, buf13, buf15, 4096, 98, grid=grid(4096), stream=stream0)
        buf14 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf13, buf14, 1024, 4, grid=grid(1024), stream=stream0)
        buf16 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf18 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf15, squeeze_310, buf16, buf18, 1024, 4, grid=grid(1024), stream=stream0)
        buf17 = buf9; del buf9  # reuse
        buf19 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6.run(buf19, le_1, le, buf2, buf11, convolution_103, unsqueeze_434, buf16, squeeze_310, buf14, primals_311, 392, 1024, grid=grid(392, 1024), stream=stream0)
        del convolution_103
        del primals_311
        del squeeze_310
        del unsqueeze_434
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf20 = aten.convolution_backward(buf19, relu_98, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_310
        buf21 = buf20[0]
        buf22 = buf20[1]
        del buf20
        buf23 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_7.run(relu_98, buf21, convolution_102, unsqueeze_446, buf23, buf25, 2048, 98, grid=grid(2048), stream=stream0)
        buf24 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf23, buf24, 512, 4, grid=grid(512), stream=stream0)
        buf26 = empty((512, ), device='cuda', dtype=torch.float32)
        buf27 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(buf25, squeeze_307, buf26, buf27, 512, 4, grid=grid(512), stream=stream0)
        buf28 = empty_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(relu_98, buf21, convolution_102, unsqueeze_446, buf26, squeeze_307, buf24, primals_308, buf28, 392, 512, grid=grid(392, 512), stream=stream0)
        del buf21
        del convolution_102
        del primals_308
        del relu_98
        del squeeze_307
        del unsqueeze_446
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf29 = aten.convolution_backward(buf28, relu_97, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_307
        buf30 = buf29[0]
        buf31 = buf29[1]
        del buf29
        buf32 = buf25; del buf25  # reuse
        buf34 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_7.run(relu_97, buf30, convolution_101, unsqueeze_458, buf32, buf34, 2048, 98, grid=grid(2048), stream=stream0)
        buf33 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf32, buf33, 512, 4, grid=grid(512), stream=stream0)
        buf35 = empty((512, ), device='cuda', dtype=torch.float32)
        buf36 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(buf34, squeeze_304, buf35, buf36, 512, 4, grid=grid(512), stream=stream0)
        buf37 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(relu_97, buf30, convolution_101, unsqueeze_458, buf35, squeeze_304, buf33, primals_305, buf37, 392, 512, grid=grid(392, 512), stream=stream0)
        del buf30
        del convolution_101
        del primals_305
        del relu_97
        del squeeze_304
        del unsqueeze_458
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf38 = aten.convolution_backward(buf37, relu_96, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_304
        buf39 = buf38[0]
        buf40 = buf38[1]
        del buf38
        buf41 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]
        triton_poi_fused_add_div_threshold_backward_11.run(buf41, relu_96, buf11, le_1, le, buf2, 8192, 49, grid=grid(8192, 49), stream=stream0)
        del buf2
        del le
        del le_1
        del relu_96
        buf42 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_12.run(buf41, buf42, 1024, 392, grid=grid(1024), stream=stream0)
        buf43 = buf15; del buf15  # reuse
        buf68 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_13.run(buf41, convolution_100, unsqueeze_470, convolution_97, unsqueeze_506, buf43, buf68, 4096, 98, grid=grid(4096), stream=stream0)
        buf44 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf45 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf43, squeeze_301, buf44, buf45, 1024, 4, grid=grid(1024), stream=stream0)
        del buf43
        buf69 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf70 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf68, squeeze_292, buf69, buf70, 1024, 4, grid=grid(1024), stream=stream0)
        del buf68
        buf46 = buf19; del buf19  # reuse
        buf71 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_14.run(buf41, convolution_100, unsqueeze_470, buf44, squeeze_301, buf42, primals_302, convolution_97, unsqueeze_506, buf69, squeeze_292, primals_293, buf46, buf71, 392, 1024, grid=grid(392, 1024), stream=stream0)
        del buf41
        del buf44
        del buf69
        del convolution_100
        del convolution_97
        del primals_293
        del primals_302
        del squeeze_292
        del squeeze_301
        del unsqueeze_470
        del unsqueeze_506
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf47 = aten.convolution_backward(buf46, relu_95, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf46
        del primals_301
        buf48 = buf47[0]
        buf49 = buf47[1]
        del buf47
        buf50 = buf34; del buf34  # reuse
        buf52 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_7.run(relu_95, buf48, convolution_99, unsqueeze_482, buf50, buf52, 2048, 98, grid=grid(2048), stream=stream0)
        buf51 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf50, buf51, 512, 4, grid=grid(512), stream=stream0)
        del buf50
        buf53 = empty((512, ), device='cuda', dtype=torch.float32)
        buf54 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(buf52, squeeze_298, buf53, buf54, 512, 4, grid=grid(512), stream=stream0)
        del buf52
        buf55 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(relu_95, buf48, convolution_99, unsqueeze_482, buf53, squeeze_298, buf51, primals_299, buf55, 392, 512, grid=grid(392, 512), stream=stream0)
        del buf48
        del convolution_99
        del primals_299
        del relu_95
        del squeeze_298
        del unsqueeze_482
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf56 = aten.convolution_backward(buf55, relu_94, primals_298, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf55
        del primals_298
        buf57 = buf56[0]
        buf58 = buf56[1]
        del buf56
        buf59 = empty((512, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_15.run(relu_94, buf57, buf59, 6656, 121, grid=grid(6656), stream=stream0)
        buf60 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_16.run(buf59, buf60, 512, 13, grid=grid(512), stream=stream0)
        buf61 = reinterpret_tensor(buf59, (512, 13), (1, 512), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_94, buf57, convolution_98, unsqueeze_494, buf61, 6656, 121, grid=grid(6656), stream=stream0)
        buf62 = empty((512, ), device='cuda', dtype=torch.float32)
        buf63 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf61, squeeze_295, buf62, buf63, 512, 13, grid=grid(512), stream=stream0)
        buf64 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(relu_94, buf57, convolution_98, unsqueeze_494, buf62, squeeze_295, buf60, primals_296, buf64, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del buf57
        del convolution_98
        del primals_296
        del relu_94
        del squeeze_295
        del unsqueeze_494
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf65 = aten.convolution_backward(buf64, relu_93, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_295
        buf66 = buf65[0]
        buf67 = buf65[1]
        del buf65
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf72 = aten.convolution_backward(buf71, getitem_210, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_210
        del primals_292
        buf73 = buf72[0]
        buf74 = buf72[1]
        del buf72
        buf75 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_20.run(buf75, buf11, 200704, grid=grid(200704), stream=stream0)
        del buf11
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        buf76 = aten.max_pool2d_with_indices_backward(buf75, relu_93, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_211)
        del buf75
        del getitem_211
        buf77 = buf76
        del buf76
        buf78 = buf61; del buf61  # reuse
        buf80 = empty_strided((512, 13), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_93, buf66, buf77, convolution_96, unsqueeze_518, buf78, buf80, 6656, 121, grid=grid(6656), stream=stream0)
        buf79 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf78, buf79, 512, 13, grid=grid(512), stream=stream0)
        buf81 = empty((512, ), device='cuda', dtype=torch.float32)
        buf83 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf80, squeeze_289, buf81, buf83, 512, 13, grid=grid(512), stream=stream0)
        buf82 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_23.run(relu_93, buf66, buf77, convolution_96, unsqueeze_518, buf81, squeeze_289, buf79, primals_290, buf82, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_96
        del primals_290
        del squeeze_289
        del unsqueeze_518
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf84 = aten.convolution_backward(buf82, cat_12, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf82
        del cat_12
        del primals_289
        buf85 = buf84[0]
        buf86 = buf84[1]
        del buf84
        buf87 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_24.run(buf87, le_8, relu_93, buf66, buf85, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del le_8
        del relu_93
        buf88 = buf80; del buf80  # reuse
        buf90 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_25.run(buf87, convolution_95, unsqueeze_530, buf88, buf90, 6656, 121, grid=grid(6656), stream=stream0)
        buf89 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf88, buf89, 512, 13, grid=grid(512), stream=stream0)
        buf91 = empty((512, ), device='cuda', dtype=torch.float32)
        buf92 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf90, squeeze_286, buf91, buf92, 512, 13, grid=grid(512), stream=stream0)
        buf93 = reinterpret_tensor(buf66, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_26.run(buf87, convolution_95, unsqueeze_530, buf91, squeeze_286, buf89, primals_287, buf93, 802816, grid=grid(802816), stream=stream0)
        del convolution_95
        del primals_287
        del squeeze_286
        del unsqueeze_530
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf94 = aten.convolution_backward(buf93, relu_91, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_286
        buf95 = buf94[0]
        buf96 = buf94[1]
        del buf94
        buf97 = empty((256, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_91, buf95, buf97, 3328, 121, grid=grid(3328), stream=stream0)
        buf98 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf97, buf98, 256, 13, grid=grid(256), stream=stream0)
        buf99 = reinterpret_tensor(buf97, (256, 13), (1, 256), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_91, buf95, convolution_94, unsqueeze_542, buf99, 3328, 121, grid=grid(3328), stream=stream0)
        buf100 = empty((256, ), device='cuda', dtype=torch.float32)
        buf101 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf99, squeeze_283, buf100, buf101, 256, 13, grid=grid(256), stream=stream0)
        buf102 = reinterpret_tensor(buf71, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_91, buf95, convolution_94, unsqueeze_542, buf100, squeeze_283, buf98, primals_284, buf102, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf95
        del convolution_94
        del primals_284
        del relu_91
        del squeeze_283
        del unsqueeze_542
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf103 = aten.convolution_backward(buf102, relu_90, primals_283, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_283
        buf104 = buf103[0]
        buf105 = buf103[1]
        del buf103
        buf106 = reinterpret_tensor(buf99, (256, 13), (13, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_90, buf104, buf106, 3328, 121, grid=grid(3328), stream=stream0)
        buf107 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf106, buf107, 256, 13, grid=grid(256), stream=stream0)
        buf108 = reinterpret_tensor(buf106, (256, 13), (1, 256), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_90, buf104, convolution_93, unsqueeze_554, buf108, 3328, 121, grid=grid(3328), stream=stream0)
        buf109 = empty((256, ), device='cuda', dtype=torch.float32)
        buf110 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf108, squeeze_280, buf109, buf110, 256, 13, grid=grid(256), stream=stream0)
        buf111 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_90, buf104, convolution_93, unsqueeze_554, buf109, squeeze_280, buf107, primals_281, buf111, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf104
        del convolution_93
        del primals_281
        del relu_90
        del squeeze_280
        del unsqueeze_554
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf112 = aten.convolution_backward(buf111, relu_89, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_280
        buf113 = buf112[0]
        buf114 = buf112[1]
        del buf112
        buf115 = reinterpret_tensor(buf90, (512, 13), (13, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_32.run(relu_89, buf85, buf87, buf113, buf115, 6656, 121, grid=grid(6656), stream=stream0)
        buf116 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_16.run(buf115, buf116, 512, 13, grid=grid(512), stream=stream0)
        buf117 = reinterpret_tensor(buf115, (512, 13), (1, 512), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_33.run(relu_89, buf85, buf87, buf113, convolution_92, unsqueeze_566, buf117, 6656, 121, grid=grid(6656), stream=stream0)
        buf118 = empty((512, ), device='cuda', dtype=torch.float32)
        buf120 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf117, squeeze_277, buf118, buf120, 512, 13, grid=grid(512), stream=stream0)
        buf119 = buf93; del buf93  # reuse
        buf121 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_34.run(buf121, relu_89, buf85, buf87, buf113, convolution_92, unsqueeze_566, buf118, squeeze_277, buf116, primals_278, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_92
        del primals_278
        del squeeze_277
        del unsqueeze_566
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf122 = aten.convolution_backward(buf121, relu_88, primals_277, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf121
        del primals_277
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        buf125 = reinterpret_tensor(buf108, (256, 13), (13, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_88, buf123, buf125, 3328, 121, grid=grid(3328), stream=stream0)
        buf126 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf125, buf126, 256, 13, grid=grid(256), stream=stream0)
        buf127 = reinterpret_tensor(buf125, (256, 13), (1, 256), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_88, buf123, convolution_91, unsqueeze_578, buf127, 3328, 121, grid=grid(3328), stream=stream0)
        buf128 = empty((256, ), device='cuda', dtype=torch.float32)
        buf129 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf127, squeeze_274, buf128, buf129, 256, 13, grid=grid(256), stream=stream0)
        buf130 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_88, buf123, convolution_91, unsqueeze_578, buf128, squeeze_274, buf126, primals_275, buf130, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf123
        del convolution_91
        del primals_275
        del relu_88
        del squeeze_274
        del unsqueeze_578
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf131 = aten.convolution_backward(buf130, relu_87, primals_274, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_274
        buf132 = buf131[0]
        buf133 = buf131[1]
        del buf131
        buf134 = reinterpret_tensor(buf127, (256, 13), (13, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_87, buf132, buf134, 3328, 121, grid=grid(3328), stream=stream0)
        buf135 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf134, buf135, 256, 13, grid=grid(256), stream=stream0)
        buf136 = reinterpret_tensor(buf134, (256, 13), (1, 256), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_87, buf132, convolution_90, unsqueeze_590, buf136, 3328, 121, grid=grid(3328), stream=stream0)
        buf137 = empty((256, ), device='cuda', dtype=torch.float32)
        buf138 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf136, squeeze_271, buf137, buf138, 256, 13, grid=grid(256), stream=stream0)
        buf139 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_87, buf132, convolution_90, unsqueeze_590, buf137, squeeze_271, buf135, primals_272, buf139, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf132
        del convolution_90
        del primals_272
        del relu_87
        del squeeze_271
        del unsqueeze_590
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf140 = aten.convolution_backward(buf139, relu_86, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_271
        buf141 = buf140[0]
        buf142 = buf140[1]
        del buf140
        buf143 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_35.run(buf143, relu_86, buf85, relu_89, buf87, buf141, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf141
        del relu_86
        del relu_89
        buf144 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf143, buf144, 512, 1568, grid=grid(512), stream=stream0)
        buf145 = reinterpret_tensor(buf117, (512, 13), (13, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf143, convolution_89, unsqueeze_602, buf145, 6656, 121, grid=grid(6656), stream=stream0)
        buf146 = empty((512, ), device='cuda', dtype=torch.float32)
        buf147 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf145, squeeze_268, buf146, buf147, 512, 13, grid=grid(512), stream=stream0)
        buf148 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf143, convolution_89, unsqueeze_602, buf146, squeeze_268, buf144, primals_269, buf148, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_89
        del primals_269
        del squeeze_268
        del unsqueeze_602
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf149 = aten.convolution_backward(buf148, cat_11, primals_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_11
        del primals_268
        buf150 = buf149[0]
        buf151 = buf149[1]
        del buf149
        buf152 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_40.run(le_15, buf143, buf150, buf152, 512, 1568, grid=grid(512), stream=stream0)
        buf153 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_41.run(le_15, buf143, buf150, convolution_88, unsqueeze_614, buf153, 6656, 121, grid=grid(6656), stream=stream0)
        buf154 = empty((512, ), device='cuda', dtype=torch.float32)
        buf156 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf153, squeeze_265, buf154, buf156, 512, 13, grid=grid(512), stream=stream0)
        buf155 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42.run(le_15, buf143, buf150, convolution_88, unsqueeze_614, buf154, squeeze_265, buf152, primals_266, buf155, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_88
        del primals_266
        del squeeze_265
        del unsqueeze_614
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf157 = aten.convolution_backward(buf155, relu_84, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf155
        del primals_265
        buf158 = buf157[0]
        buf159 = buf157[1]
        del buf157
        buf160 = reinterpret_tensor(buf136, (256, 13), (13, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_84, buf158, buf160, 3328, 121, grid=grid(3328), stream=stream0)
        buf161 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf160, buf161, 256, 13, grid=grid(256), stream=stream0)
        buf162 = reinterpret_tensor(buf160, (256, 13), (1, 256), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_84, buf158, convolution_87, unsqueeze_626, buf162, 3328, 121, grid=grid(3328), stream=stream0)
        buf163 = empty((256, ), device='cuda', dtype=torch.float32)
        buf164 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf162, squeeze_262, buf163, buf164, 256, 13, grid=grid(256), stream=stream0)
        buf165 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_84, buf158, convolution_87, unsqueeze_626, buf163, squeeze_262, buf161, primals_263, buf165, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf158
        del convolution_87
        del primals_263
        del relu_84
        del squeeze_262
        del unsqueeze_626
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf166 = aten.convolution_backward(buf165, relu_83, primals_262, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_262
        buf167 = buf166[0]
        buf168 = buf166[1]
        del buf166
        buf169 = reinterpret_tensor(buf162, (256, 13), (13, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_83, buf167, buf169, 3328, 121, grid=grid(3328), stream=stream0)
        buf170 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf169, buf170, 256, 13, grid=grid(256), stream=stream0)
        buf171 = reinterpret_tensor(buf169, (256, 13), (1, 256), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_83, buf167, convolution_86, unsqueeze_638, buf171, 3328, 121, grid=grid(3328), stream=stream0)
        buf172 = empty((256, ), device='cuda', dtype=torch.float32)
        buf173 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf171, squeeze_259, buf172, buf173, 256, 13, grid=grid(256), stream=stream0)
        buf174 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_83, buf167, convolution_86, unsqueeze_638, buf172, squeeze_259, buf170, primals_260, buf174, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf167
        del convolution_86
        del primals_260
        del relu_83
        del squeeze_259
        del unsqueeze_638
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf175 = aten.convolution_backward(buf174, relu_82, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_259
        buf176 = buf175[0]
        buf177 = buf175[1]
        del buf175
        buf178 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_43.run(buf178, relu_82, buf150, le_15, buf176, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf150
        del le_15
        del relu_82
        buf179 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf178, buf179, 512, 1568, grid=grid(512), stream=stream0)
        buf180 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf178, convolution_85, unsqueeze_650, buf180, 6656, 121, grid=grid(6656), stream=stream0)
        buf181 = empty((512, ), device='cuda', dtype=torch.float32)
        buf182 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf180, squeeze_256, buf181, buf182, 512, 13, grid=grid(512), stream=stream0)
        buf183 = reinterpret_tensor(buf176, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf178, convolution_85, unsqueeze_650, buf181, squeeze_256, buf179, primals_257, buf183, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_85
        del primals_257
        del squeeze_256
        del unsqueeze_650
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf184 = aten.convolution_backward(buf183, relu_81, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_256
        buf185 = buf184[0]
        buf186 = buf184[1]
        del buf184
        buf187 = reinterpret_tensor(buf171, (256, 13), (13, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_81, buf185, buf187, 3328, 121, grid=grid(3328), stream=stream0)
        buf188 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf187, buf188, 256, 13, grid=grid(256), stream=stream0)
        buf189 = reinterpret_tensor(buf187, (256, 13), (1, 256), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_81, buf185, convolution_84, unsqueeze_662, buf189, 3328, 121, grid=grid(3328), stream=stream0)
        buf190 = empty((256, ), device='cuda', dtype=torch.float32)
        buf191 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf189, squeeze_253, buf190, buf191, 256, 13, grid=grid(256), stream=stream0)
        buf192 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_81, buf185, convolution_84, unsqueeze_662, buf190, squeeze_253, buf188, primals_254, buf192, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf185
        del convolution_84
        del primals_254
        del relu_81
        del squeeze_253
        del unsqueeze_662
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf193 = aten.convolution_backward(buf192, relu_80, primals_253, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_253
        buf194 = buf193[0]
        buf195 = buf193[1]
        del buf193
        buf196 = reinterpret_tensor(buf189, (256, 13), (13, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_80, buf194, buf196, 3328, 121, grid=grid(3328), stream=stream0)
        buf197 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf196, buf197, 256, 13, grid=grid(256), stream=stream0)
        buf198 = reinterpret_tensor(buf196, (256, 13), (1, 256), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_80, buf194, convolution_83, unsqueeze_674, buf198, 3328, 121, grid=grid(3328), stream=stream0)
        buf199 = empty((256, ), device='cuda', dtype=torch.float32)
        buf200 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf198, squeeze_250, buf199, buf200, 256, 13, grid=grid(256), stream=stream0)
        buf201 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_80, buf194, convolution_83, unsqueeze_674, buf199, squeeze_250, buf197, primals_251, buf201, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf194
        del convolution_83
        del primals_251
        del relu_80
        del squeeze_250
        del unsqueeze_674
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf202 = aten.convolution_backward(buf201, relu_79, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_250
        buf203 = buf202[0]
        buf204 = buf202[1]
        del buf202
        buf205 = buf181; del buf181  # reuse
        buf206 = empty((512, ), device='cuda', dtype=torch.float32)
        buf208 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(relu_79, buf85, buf178, buf203, convolution_82, unsqueeze_686, squeeze_247, buf205, buf206, buf208, 512, 1568, grid=grid(512), stream=stream0)
        buf209 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(relu_79, buf85, buf178, buf203, convolution_82, unsqueeze_686, buf206, squeeze_247, buf205, primals_248, buf209, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del convolution_82
        del primals_248
        del squeeze_247
        del unsqueeze_686
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf210 = aten.convolution_backward(buf209, cat_10, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf209
        del cat_10
        del primals_247
        buf211 = buf210[0]
        buf212 = buf210[1]
        del buf210
        buf213 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_46.run(buf213, le_22, relu_79, buf85, buf203, buf211, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del le_22
        del relu_79
        buf214 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf213, buf214, 512, 1568, grid=grid(512), stream=stream0)
        buf215 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf213, convolution_81, unsqueeze_698, buf215, 6656, 121, grid=grid(6656), stream=stream0)
        buf216 = empty((512, ), device='cuda', dtype=torch.float32)
        buf217 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf215, squeeze_244, buf216, buf217, 512, 13, grid=grid(512), stream=stream0)
        buf218 = reinterpret_tensor(buf203, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf213, convolution_81, unsqueeze_698, buf216, squeeze_244, buf214, primals_245, buf218, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_81
        del primals_245
        del squeeze_244
        del unsqueeze_698
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf219 = aten.convolution_backward(buf218, relu_77, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_244
        buf220 = buf219[0]
        buf221 = buf219[1]
        del buf219
        buf222 = reinterpret_tensor(buf198, (256, 13), (13, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_77, buf220, buf222, 3328, 121, grid=grid(3328), stream=stream0)
        buf223 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf222, buf223, 256, 13, grid=grid(256), stream=stream0)
        buf224 = reinterpret_tensor(buf222, (256, 13), (1, 256), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_77, buf220, convolution_80, unsqueeze_710, buf224, 3328, 121, grid=grid(3328), stream=stream0)
        buf225 = empty((256, ), device='cuda', dtype=torch.float32)
        buf226 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf224, squeeze_241, buf225, buf226, 256, 13, grid=grid(256), stream=stream0)
        buf227 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_77, buf220, convolution_80, unsqueeze_710, buf225, squeeze_241, buf223, primals_242, buf227, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf220
        del convolution_80
        del primals_242
        del relu_77
        del squeeze_241
        del unsqueeze_710
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf228 = aten.convolution_backward(buf227, relu_76, primals_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_241
        buf229 = buf228[0]
        buf230 = buf228[1]
        del buf228
        buf231 = reinterpret_tensor(buf224, (256, 13), (13, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_76, buf229, buf231, 3328, 121, grid=grid(3328), stream=stream0)
        buf232 = buf225; del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf231, buf232, 256, 13, grid=grid(256), stream=stream0)
        buf233 = reinterpret_tensor(buf231, (256, 13), (1, 256), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_76, buf229, convolution_79, unsqueeze_722, buf233, 3328, 121, grid=grid(3328), stream=stream0)
        buf234 = empty((256, ), device='cuda', dtype=torch.float32)
        buf235 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf233, squeeze_238, buf234, buf235, 256, 13, grid=grid(256), stream=stream0)
        buf236 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_76, buf229, convolution_79, unsqueeze_722, buf234, squeeze_238, buf232, primals_239, buf236, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf229
        del convolution_79
        del primals_239
        del relu_76
        del squeeze_238
        del unsqueeze_722
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf237 = aten.convolution_backward(buf236, relu_75, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_238
        buf238 = buf237[0]
        buf239 = buf237[1]
        del buf237
        buf240 = buf216; del buf216  # reuse
        buf241 = empty((512, ), device='cuda', dtype=torch.float32)
        buf243 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_47.run(relu_75, buf211, buf213, buf238, convolution_78, unsqueeze_734, squeeze_235, buf240, buf241, buf243, 512, 1568, grid=grid(512), stream=stream0)
        buf244 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_48.run(relu_75, buf211, buf213, buf238, convolution_78, unsqueeze_734, buf241, squeeze_235, buf240, primals_236, buf244, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del convolution_78
        del primals_236
        del squeeze_235
        del unsqueeze_734
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf245 = aten.convolution_backward(buf244, relu_74, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf244
        del primals_235
        buf246 = buf245[0]
        buf247 = buf245[1]
        del buf245
        buf248 = reinterpret_tensor(buf233, (256, 13), (13, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_74, buf246, buf248, 3328, 121, grid=grid(3328), stream=stream0)
        buf249 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf248, buf249, 256, 13, grid=grid(256), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (256, 13), (1, 256), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_74, buf246, convolution_77, unsqueeze_746, buf250, 3328, 121, grid=grid(3328), stream=stream0)
        buf251 = empty((256, ), device='cuda', dtype=torch.float32)
        buf252 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf250, squeeze_232, buf251, buf252, 256, 13, grid=grid(256), stream=stream0)
        buf253 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_74, buf246, convolution_77, unsqueeze_746, buf251, squeeze_232, buf249, primals_233, buf253, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf246
        del convolution_77
        del primals_233
        del relu_74
        del squeeze_232
        del unsqueeze_746
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf254 = aten.convolution_backward(buf253, relu_73, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_232
        buf255 = buf254[0]
        buf256 = buf254[1]
        del buf254
        buf257 = reinterpret_tensor(buf250, (256, 13), (13, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_73, buf255, buf257, 3328, 121, grid=grid(3328), stream=stream0)
        buf258 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf257, buf258, 256, 13, grid=grid(256), stream=stream0)
        buf259 = reinterpret_tensor(buf257, (256, 13), (1, 256), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_73, buf255, convolution_76, unsqueeze_758, buf259, 3328, 121, grid=grid(3328), stream=stream0)
        buf260 = empty((256, ), device='cuda', dtype=torch.float32)
        buf261 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf259, squeeze_229, buf260, buf261, 256, 13, grid=grid(256), stream=stream0)
        buf262 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_73, buf255, convolution_76, unsqueeze_758, buf260, squeeze_229, buf258, primals_230, buf262, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf255
        del convolution_76
        del primals_230
        del relu_73
        del squeeze_229
        del unsqueeze_758
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf263 = aten.convolution_backward(buf262, relu_72, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_229
        buf264 = buf263[0]
        buf265 = buf263[1]
        del buf263
        buf266 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_49.run(buf266, relu_72, buf211, relu_75, buf238, buf264, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf211
        del buf238
        del relu_72
        del relu_75
        buf267 = buf241; del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf266, buf267, 512, 1568, grid=grid(512), stream=stream0)
        buf268 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf266, convolution_75, unsqueeze_770, buf268, 6656, 121, grid=grid(6656), stream=stream0)
        buf269 = empty((512, ), device='cuda', dtype=torch.float32)
        buf270 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf268, squeeze_226, buf269, buf270, 512, 13, grid=grid(512), stream=stream0)
        buf271 = reinterpret_tensor(buf264, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf266, convolution_75, unsqueeze_770, buf269, squeeze_226, buf267, primals_227, buf271, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_75
        del primals_227
        del squeeze_226
        del unsqueeze_770
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf272 = aten.convolution_backward(buf271, cat_9, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_9
        del primals_226
        buf273 = buf272[0]
        buf274 = buf272[1]
        del buf272
        buf275 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_40.run(le_29, buf266, buf273, buf275, 512, 1568, grid=grid(512), stream=stream0)
        buf276 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_41.run(le_29, buf266, buf273, convolution_74, unsqueeze_782, buf276, 6656, 121, grid=grid(6656), stream=stream0)
        buf277 = empty((512, ), device='cuda', dtype=torch.float32)
        buf279 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf276, squeeze_223, buf277, buf279, 512, 13, grid=grid(512), stream=stream0)
        buf278 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42.run(le_29, buf266, buf273, convolution_74, unsqueeze_782, buf277, squeeze_223, buf275, primals_224, buf278, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_74
        del primals_224
        del squeeze_223
        del unsqueeze_782
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf280 = aten.convolution_backward(buf278, relu_70, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf278
        del primals_223
        buf281 = buf280[0]
        buf282 = buf280[1]
        del buf280
        buf283 = reinterpret_tensor(buf259, (256, 13), (13, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_70, buf281, buf283, 3328, 121, grid=grid(3328), stream=stream0)
        buf284 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf283, buf284, 256, 13, grid=grid(256), stream=stream0)
        buf285 = reinterpret_tensor(buf283, (256, 13), (1, 256), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_70, buf281, convolution_73, unsqueeze_794, buf285, 3328, 121, grid=grid(3328), stream=stream0)
        buf286 = empty((256, ), device='cuda', dtype=torch.float32)
        buf287 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf285, squeeze_220, buf286, buf287, 256, 13, grid=grid(256), stream=stream0)
        buf288 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_70, buf281, convolution_73, unsqueeze_794, buf286, squeeze_220, buf284, primals_221, buf288, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf281
        del convolution_73
        del primals_221
        del relu_70
        del squeeze_220
        del unsqueeze_794
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf289 = aten.convolution_backward(buf288, relu_69, primals_220, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_220
        buf290 = buf289[0]
        buf291 = buf289[1]
        del buf289
        buf292 = reinterpret_tensor(buf285, (256, 13), (13, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_69, buf290, buf292, 3328, 121, grid=grid(3328), stream=stream0)
        buf293 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf292, buf293, 256, 13, grid=grid(256), stream=stream0)
        buf294 = reinterpret_tensor(buf292, (256, 13), (1, 256), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_69, buf290, convolution_72, unsqueeze_806, buf294, 3328, 121, grid=grid(3328), stream=stream0)
        buf295 = empty((256, ), device='cuda', dtype=torch.float32)
        buf296 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf294, squeeze_217, buf295, buf296, 256, 13, grid=grid(256), stream=stream0)
        buf297 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_69, buf290, convolution_72, unsqueeze_806, buf295, squeeze_217, buf293, primals_218, buf297, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf290
        del convolution_72
        del primals_218
        del relu_69
        del squeeze_217
        del unsqueeze_806
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf298 = aten.convolution_backward(buf297, relu_68, primals_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_217
        buf299 = buf298[0]
        buf300 = buf298[1]
        del buf298
        buf301 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_43.run(buf301, relu_68, buf273, le_29, buf299, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf273
        del le_29
        del relu_68
        buf302 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf301, buf302, 512, 1568, grid=grid(512), stream=stream0)
        buf303 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf301, convolution_71, unsqueeze_818, buf303, 6656, 121, grid=grid(6656), stream=stream0)
        buf304 = empty((512, ), device='cuda', dtype=torch.float32)
        buf305 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf303, squeeze_214, buf304, buf305, 512, 13, grid=grid(512), stream=stream0)
        buf306 = reinterpret_tensor(buf299, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf301, convolution_71, unsqueeze_818, buf304, squeeze_214, buf302, primals_215, buf306, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_71
        del primals_215
        del squeeze_214
        del unsqueeze_818
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf307 = aten.convolution_backward(buf306, relu_67, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_214
        buf308 = buf307[0]
        buf309 = buf307[1]
        del buf307
        buf310 = reinterpret_tensor(buf294, (256, 13), (13, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_67, buf308, buf310, 3328, 121, grid=grid(3328), stream=stream0)
        buf311 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf310, buf311, 256, 13, grid=grid(256), stream=stream0)
        buf312 = reinterpret_tensor(buf310, (256, 13), (1, 256), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_67, buf308, convolution_70, unsqueeze_830, buf312, 3328, 121, grid=grid(3328), stream=stream0)
        buf313 = empty((256, ), device='cuda', dtype=torch.float32)
        buf314 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf312, squeeze_211, buf313, buf314, 256, 13, grid=grid(256), stream=stream0)
        buf315 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_67, buf308, convolution_70, unsqueeze_830, buf313, squeeze_211, buf311, primals_212, buf315, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf308
        del convolution_70
        del primals_212
        del relu_67
        del squeeze_211
        del unsqueeze_830
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf316 = aten.convolution_backward(buf315, relu_66, primals_211, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_211
        buf317 = buf316[0]
        buf318 = buf316[1]
        del buf316
        buf319 = reinterpret_tensor(buf312, (256, 13), (13, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_66, buf317, buf319, 3328, 121, grid=grid(3328), stream=stream0)
        buf320 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf319, buf320, 256, 13, grid=grid(256), stream=stream0)
        buf321 = reinterpret_tensor(buf319, (256, 13), (1, 256), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_66, buf317, convolution_69, unsqueeze_842, buf321, 3328, 121, grid=grid(3328), stream=stream0)
        buf322 = empty((256, ), device='cuda', dtype=torch.float32)
        buf323 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf321, squeeze_208, buf322, buf323, 256, 13, grid=grid(256), stream=stream0)
        buf324 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_66, buf317, convolution_69, unsqueeze_842, buf322, squeeze_208, buf320, primals_209, buf324, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf317
        del convolution_69
        del primals_209
        del relu_66
        del squeeze_208
        del unsqueeze_842
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf325 = aten.convolution_backward(buf324, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_208
        buf326 = buf325[0]
        buf327 = buf325[1]
        del buf325
        buf328 = buf304; del buf304  # reuse
        buf329 = empty((512, ), device='cuda', dtype=torch.float32)
        buf331 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_50.run(relu_65, buf85, buf301, buf326, convolution_68, unsqueeze_854, squeeze_205, buf328, buf329, buf331, 512, 1568, grid=grid(512), stream=stream0)
        buf332 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_51.run(relu_65, buf85, buf301, buf326, convolution_68, unsqueeze_854, buf329, squeeze_205, buf328, primals_206, buf332, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del convolution_68
        del primals_206
        del squeeze_205
        del unsqueeze_854
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf333 = aten.convolution_backward(buf332, cat_8, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf332
        del cat_8
        del primals_205
        buf334 = buf333[0]
        buf335 = buf333[1]
        del buf333
        buf336 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_52.run(buf336, le_36, relu_65, buf85, buf326, buf334, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del le_36
        del relu_65
        buf337 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf336, buf337, 512, 1568, grid=grid(512), stream=stream0)
        buf338 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf336, convolution_67, unsqueeze_866, buf338, 6656, 121, grid=grid(6656), stream=stream0)
        buf339 = empty((512, ), device='cuda', dtype=torch.float32)
        buf340 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf338, squeeze_202, buf339, buf340, 512, 13, grid=grid(512), stream=stream0)
        buf341 = reinterpret_tensor(buf326, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf336, convolution_67, unsqueeze_866, buf339, squeeze_202, buf337, primals_203, buf341, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_67
        del primals_203
        del squeeze_202
        del unsqueeze_866
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf342 = aten.convolution_backward(buf341, relu_63, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_202
        buf343 = buf342[0]
        buf344 = buf342[1]
        del buf342
        buf345 = reinterpret_tensor(buf321, (256, 13), (13, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_63, buf343, buf345, 3328, 121, grid=grid(3328), stream=stream0)
        buf346 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf345, buf346, 256, 13, grid=grid(256), stream=stream0)
        buf347 = reinterpret_tensor(buf345, (256, 13), (1, 256), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_63, buf343, convolution_66, unsqueeze_878, buf347, 3328, 121, grid=grid(3328), stream=stream0)
        buf348 = empty((256, ), device='cuda', dtype=torch.float32)
        buf349 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf347, squeeze_199, buf348, buf349, 256, 13, grid=grid(256), stream=stream0)
        buf350 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_63, buf343, convolution_66, unsqueeze_878, buf348, squeeze_199, buf346, primals_200, buf350, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf343
        del convolution_66
        del primals_200
        del relu_63
        del squeeze_199
        del unsqueeze_878
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf351 = aten.convolution_backward(buf350, relu_62, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_199
        buf352 = buf351[0]
        buf353 = buf351[1]
        del buf351
        buf354 = reinterpret_tensor(buf347, (256, 13), (13, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_62, buf352, buf354, 3328, 121, grid=grid(3328), stream=stream0)
        buf355 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf354, buf355, 256, 13, grid=grid(256), stream=stream0)
        buf356 = reinterpret_tensor(buf354, (256, 13), (1, 256), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_62, buf352, convolution_65, unsqueeze_890, buf356, 3328, 121, grid=grid(3328), stream=stream0)
        buf357 = empty((256, ), device='cuda', dtype=torch.float32)
        buf358 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf356, squeeze_196, buf357, buf358, 256, 13, grid=grid(256), stream=stream0)
        buf359 = buf350; del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_62, buf352, convolution_65, unsqueeze_890, buf357, squeeze_196, buf355, primals_197, buf359, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf352
        del convolution_65
        del primals_197
        del relu_62
        del squeeze_196
        del unsqueeze_890
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf360 = aten.convolution_backward(buf359, relu_61, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_196
        buf361 = buf360[0]
        buf362 = buf360[1]
        del buf360
        buf363 = buf339; del buf339  # reuse
        buf364 = empty((512, ), device='cuda', dtype=torch.float32)
        buf366 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_53.run(relu_61, buf334, buf336, buf361, convolution_64, unsqueeze_902, squeeze_193, buf363, buf364, buf366, 512, 1568, grid=grid(512), stream=stream0)
        buf367 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_54.run(relu_61, buf334, buf336, buf361, convolution_64, unsqueeze_902, buf364, squeeze_193, buf363, primals_194, buf367, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del convolution_64
        del primals_194
        del squeeze_193
        del unsqueeze_902
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf368 = aten.convolution_backward(buf367, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf367
        del primals_193
        buf369 = buf368[0]
        buf370 = buf368[1]
        del buf368
        buf371 = reinterpret_tensor(buf356, (256, 13), (13, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_60, buf369, buf371, 3328, 121, grid=grid(3328), stream=stream0)
        buf372 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf371, buf372, 256, 13, grid=grid(256), stream=stream0)
        buf373 = reinterpret_tensor(buf371, (256, 13), (1, 256), 0); del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_60, buf369, convolution_63, unsqueeze_914, buf373, 3328, 121, grid=grid(3328), stream=stream0)
        buf374 = empty((256, ), device='cuda', dtype=torch.float32)
        buf375 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf373, squeeze_190, buf374, buf375, 256, 13, grid=grid(256), stream=stream0)
        buf376 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_60, buf369, convolution_63, unsqueeze_914, buf374, squeeze_190, buf372, primals_191, buf376, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf369
        del convolution_63
        del primals_191
        del relu_60
        del squeeze_190
        del unsqueeze_914
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf377 = aten.convolution_backward(buf376, relu_59, primals_190, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_190
        buf378 = buf377[0]
        buf379 = buf377[1]
        del buf377
        buf380 = reinterpret_tensor(buf373, (256, 13), (13, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_59, buf378, buf380, 3328, 121, grid=grid(3328), stream=stream0)
        buf381 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf380, buf381, 256, 13, grid=grid(256), stream=stream0)
        buf382 = reinterpret_tensor(buf380, (256, 13), (1, 256), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_59, buf378, convolution_62, unsqueeze_926, buf382, 3328, 121, grid=grid(3328), stream=stream0)
        buf383 = empty((256, ), device='cuda', dtype=torch.float32)
        buf384 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf382, squeeze_187, buf383, buf384, 256, 13, grid=grid(256), stream=stream0)
        buf385 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_59, buf378, convolution_62, unsqueeze_926, buf383, squeeze_187, buf381, primals_188, buf385, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf378
        del convolution_62
        del primals_188
        del relu_59
        del squeeze_187
        del unsqueeze_926
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf386 = aten.convolution_backward(buf385, relu_58, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_187
        buf387 = buf386[0]
        buf388 = buf386[1]
        del buf386
        buf389 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_55.run(buf389, relu_58, buf334, relu_61, buf361, buf387, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf361
        del relu_58
        del relu_61
        buf390 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf389, buf390, 512, 1568, grid=grid(512), stream=stream0)
        buf391 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf389, convolution_61, unsqueeze_938, buf391, 6656, 121, grid=grid(6656), stream=stream0)
        buf392 = empty((512, ), device='cuda', dtype=torch.float32)
        buf393 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf391, squeeze_184, buf392, buf393, 512, 13, grid=grid(512), stream=stream0)
        buf394 = reinterpret_tensor(buf387, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf389, convolution_61, unsqueeze_938, buf392, squeeze_184, buf390, primals_185, buf394, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_61
        del primals_185
        del squeeze_184
        del unsqueeze_938
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf395 = aten.convolution_backward(buf394, cat_7, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_7
        del primals_184
        buf396 = buf395[0]
        buf397 = buf395[1]
        del buf395
        buf398 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_40.run(le_43, buf389, buf396, buf398, 512, 1568, grid=grid(512), stream=stream0)
        buf399 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_41.run(le_43, buf389, buf396, convolution_60, unsqueeze_950, buf399, 6656, 121, grid=grid(6656), stream=stream0)
        buf400 = empty((512, ), device='cuda', dtype=torch.float32)
        buf402 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf399, squeeze_181, buf400, buf402, 512, 13, grid=grid(512), stream=stream0)
        buf401 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42.run(le_43, buf389, buf396, convolution_60, unsqueeze_950, buf400, squeeze_181, buf398, primals_182, buf401, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_60
        del primals_182
        del squeeze_181
        del unsqueeze_950
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf403 = aten.convolution_backward(buf401, relu_56, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf401
        del primals_181
        buf404 = buf403[0]
        buf405 = buf403[1]
        del buf403
        buf406 = reinterpret_tensor(buf382, (256, 13), (13, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_56, buf404, buf406, 3328, 121, grid=grid(3328), stream=stream0)
        buf407 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf406, buf407, 256, 13, grid=grid(256), stream=stream0)
        buf408 = reinterpret_tensor(buf406, (256, 13), (1, 256), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_56, buf404, convolution_59, unsqueeze_962, buf408, 3328, 121, grid=grid(3328), stream=stream0)
        buf409 = empty((256, ), device='cuda', dtype=torch.float32)
        buf410 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf408, squeeze_178, buf409, buf410, 256, 13, grid=grid(256), stream=stream0)
        buf411 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_56, buf404, convolution_59, unsqueeze_962, buf409, squeeze_178, buf407, primals_179, buf411, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf404
        del convolution_59
        del primals_179
        del relu_56
        del squeeze_178
        del unsqueeze_962
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf412 = aten.convolution_backward(buf411, relu_55, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_178
        buf413 = buf412[0]
        buf414 = buf412[1]
        del buf412
        buf415 = reinterpret_tensor(buf408, (256, 13), (13, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_55, buf413, buf415, 3328, 121, grid=grid(3328), stream=stream0)
        buf416 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf415, buf416, 256, 13, grid=grid(256), stream=stream0)
        buf417 = reinterpret_tensor(buf415, (256, 13), (1, 256), 0); del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_55, buf413, convolution_58, unsqueeze_974, buf417, 3328, 121, grid=grid(3328), stream=stream0)
        buf418 = empty((256, ), device='cuda', dtype=torch.float32)
        buf419 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf417, squeeze_175, buf418, buf419, 256, 13, grid=grid(256), stream=stream0)
        buf420 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_55, buf413, convolution_58, unsqueeze_974, buf418, squeeze_175, buf416, primals_176, buf420, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf413
        del convolution_58
        del primals_176
        del relu_55
        del squeeze_175
        del unsqueeze_974
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf421 = aten.convolution_backward(buf420, relu_54, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_175
        buf422 = buf421[0]
        buf423 = buf421[1]
        del buf421
        buf424 = buf389; del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_43.run(buf424, relu_54, buf396, le_43, buf422, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf396
        del le_43
        del relu_54
        buf425 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf424, buf425, 512, 1568, grid=grid(512), stream=stream0)
        buf426 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf424, convolution_57, unsqueeze_986, buf426, 6656, 121, grid=grid(6656), stream=stream0)
        buf427 = empty((512, ), device='cuda', dtype=torch.float32)
        buf428 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf426, squeeze_172, buf427, buf428, 512, 13, grid=grid(512), stream=stream0)
        buf429 = reinterpret_tensor(buf422, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf424, convolution_57, unsqueeze_986, buf427, squeeze_172, buf425, primals_173, buf429, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_57
        del primals_173
        del squeeze_172
        del unsqueeze_986
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf430 = aten.convolution_backward(buf429, relu_53, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_172
        buf431 = buf430[0]
        buf432 = buf430[1]
        del buf430
        buf433 = reinterpret_tensor(buf417, (256, 13), (13, 1), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_53, buf431, buf433, 3328, 121, grid=grid(3328), stream=stream0)
        buf434 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf433, buf434, 256, 13, grid=grid(256), stream=stream0)
        buf435 = reinterpret_tensor(buf433, (256, 13), (1, 256), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_53, buf431, convolution_56, unsqueeze_998, buf435, 3328, 121, grid=grid(3328), stream=stream0)
        buf436 = empty((256, ), device='cuda', dtype=torch.float32)
        buf437 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf435, squeeze_169, buf436, buf437, 256, 13, grid=grid(256), stream=stream0)
        buf438 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_53, buf431, convolution_56, unsqueeze_998, buf436, squeeze_169, buf434, primals_170, buf438, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf431
        del convolution_56
        del primals_170
        del relu_53
        del squeeze_169
        del unsqueeze_998
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf439 = aten.convolution_backward(buf438, relu_52, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_169
        buf440 = buf439[0]
        buf441 = buf439[1]
        del buf439
        buf442 = reinterpret_tensor(buf435, (256, 13), (13, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_52, buf440, buf442, 3328, 121, grid=grid(3328), stream=stream0)
        buf443 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf442, buf443, 256, 13, grid=grid(256), stream=stream0)
        buf444 = reinterpret_tensor(buf442, (256, 13), (1, 256), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_52, buf440, convolution_55, unsqueeze_1010, buf444, 3328, 121, grid=grid(3328), stream=stream0)
        buf445 = empty((256, ), device='cuda', dtype=torch.float32)
        buf446 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf444, squeeze_166, buf445, buf446, 256, 13, grid=grid(256), stream=stream0)
        buf447 = buf438; del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_52, buf440, convolution_55, unsqueeze_1010, buf445, squeeze_166, buf443, primals_167, buf447, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf440
        del convolution_55
        del primals_167
        del relu_52
        del squeeze_166
        del unsqueeze_1010
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf448 = aten.convolution_backward(buf447, relu_51, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_166
        buf449 = buf448[0]
        buf450 = buf448[1]
        del buf448
        buf451 = buf427; del buf427  # reuse
        buf452 = empty((512, ), device='cuda', dtype=torch.float32)
        buf454 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_56.run(relu_51, buf334, buf424, buf449, convolution_54, unsqueeze_1022, squeeze_163, buf451, buf452, buf454, 512, 1568, grid=grid(512), stream=stream0)
        buf455 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_57.run(relu_51, buf334, buf424, buf449, convolution_54, unsqueeze_1022, buf452, squeeze_163, buf451, primals_164, buf455, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del convolution_54
        del primals_164
        del squeeze_163
        del unsqueeze_1022
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf456 = aten.convolution_backward(buf455, cat_6, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf455
        del cat_6
        del primals_163
        buf457 = buf456[0]
        buf458 = buf456[1]
        del buf456
        buf459 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_58.run(buf459, le_50, relu_51, buf334, buf449, buf457, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf334
        del le_50
        del relu_51
        buf460 = buf452; del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf459, buf460, 512, 1568, grid=grid(512), stream=stream0)
        buf461 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf459, convolution_53, unsqueeze_1034, buf461, 6656, 121, grid=grid(6656), stream=stream0)
        buf462 = empty((512, ), device='cuda', dtype=torch.float32)
        buf463 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf461, squeeze_160, buf462, buf463, 512, 13, grid=grid(512), stream=stream0)
        buf464 = reinterpret_tensor(buf449, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf459, convolution_53, unsqueeze_1034, buf462, squeeze_160, buf460, primals_161, buf464, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_53
        del primals_161
        del squeeze_160
        del unsqueeze_1034
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf465 = aten.convolution_backward(buf464, relu_49, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_160
        buf466 = buf465[0]
        buf467 = buf465[1]
        del buf465
        buf468 = reinterpret_tensor(buf444, (256, 13), (13, 1), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_49, buf466, buf468, 3328, 121, grid=grid(3328), stream=stream0)
        buf469 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf468, buf469, 256, 13, grid=grid(256), stream=stream0)
        buf470 = reinterpret_tensor(buf468, (256, 13), (1, 256), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_49, buf466, convolution_52, unsqueeze_1046, buf470, 3328, 121, grid=grid(3328), stream=stream0)
        buf471 = empty((256, ), device='cuda', dtype=torch.float32)
        buf472 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf470, squeeze_157, buf471, buf472, 256, 13, grid=grid(256), stream=stream0)
        buf473 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_49, buf466, convolution_52, unsqueeze_1046, buf471, squeeze_157, buf469, primals_158, buf473, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf466
        del convolution_52
        del primals_158
        del relu_49
        del squeeze_157
        del unsqueeze_1046
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf474 = aten.convolution_backward(buf473, relu_48, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_157
        buf475 = buf474[0]
        buf476 = buf474[1]
        del buf474
        buf477 = reinterpret_tensor(buf470, (256, 13), (13, 1), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_48, buf475, buf477, 3328, 121, grid=grid(3328), stream=stream0)
        buf478 = buf471; del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf477, buf478, 256, 13, grid=grid(256), stream=stream0)
        buf479 = reinterpret_tensor(buf477, (256, 13), (1, 256), 0); del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_48, buf475, convolution_51, unsqueeze_1058, buf479, 3328, 121, grid=grid(3328), stream=stream0)
        buf480 = empty((256, ), device='cuda', dtype=torch.float32)
        buf481 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf479, squeeze_154, buf480, buf481, 256, 13, grid=grid(256), stream=stream0)
        buf482 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_48, buf475, convolution_51, unsqueeze_1058, buf480, squeeze_154, buf478, primals_155, buf482, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf475
        del convolution_51
        del primals_155
        del relu_48
        del squeeze_154
        del unsqueeze_1058
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf483 = aten.convolution_backward(buf482, relu_47, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_154
        buf484 = buf483[0]
        buf485 = buf483[1]
        del buf483
        buf486 = buf462; del buf462  # reuse
        buf487 = empty((512, ), device='cuda', dtype=torch.float32)
        buf489 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_47.run(relu_47, buf457, buf459, buf484, convolution_50, unsqueeze_1070, squeeze_151, buf486, buf487, buf489, 512, 1568, grid=grid(512), stream=stream0)
        buf490 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_48.run(relu_47, buf457, buf459, buf484, convolution_50, unsqueeze_1070, buf487, squeeze_151, buf486, primals_152, buf490, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del convolution_50
        del primals_152
        del squeeze_151
        del unsqueeze_1070
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf491 = aten.convolution_backward(buf490, relu_46, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf490
        del primals_151
        buf492 = buf491[0]
        buf493 = buf491[1]
        del buf491
        buf494 = reinterpret_tensor(buf479, (256, 13), (13, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_46, buf492, buf494, 3328, 121, grid=grid(3328), stream=stream0)
        buf495 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf494, buf495, 256, 13, grid=grid(256), stream=stream0)
        buf496 = reinterpret_tensor(buf494, (256, 13), (1, 256), 0); del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_46, buf492, convolution_49, unsqueeze_1082, buf496, 3328, 121, grid=grid(3328), stream=stream0)
        buf497 = empty((256, ), device='cuda', dtype=torch.float32)
        buf498 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf496, squeeze_148, buf497, buf498, 256, 13, grid=grid(256), stream=stream0)
        buf499 = buf482; del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_46, buf492, convolution_49, unsqueeze_1082, buf497, squeeze_148, buf495, primals_149, buf499, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf492
        del convolution_49
        del primals_149
        del relu_46
        del squeeze_148
        del unsqueeze_1082
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf500 = aten.convolution_backward(buf499, relu_45, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_148
        buf501 = buf500[0]
        buf502 = buf500[1]
        del buf500
        buf503 = reinterpret_tensor(buf496, (256, 13), (13, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_45, buf501, buf503, 3328, 121, grid=grid(3328), stream=stream0)
        buf504 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf503, buf504, 256, 13, grid=grid(256), stream=stream0)
        buf505 = reinterpret_tensor(buf503, (256, 13), (1, 256), 0); del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_45, buf501, convolution_48, unsqueeze_1094, buf505, 3328, 121, grid=grid(3328), stream=stream0)
        buf506 = empty((256, ), device='cuda', dtype=torch.float32)
        buf507 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf505, squeeze_145, buf506, buf507, 256, 13, grid=grid(256), stream=stream0)
        buf508 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_45, buf501, convolution_48, unsqueeze_1094, buf506, squeeze_145, buf504, primals_146, buf508, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf501
        del convolution_48
        del primals_146
        del relu_45
        del squeeze_145
        del unsqueeze_1094
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf509 = aten.convolution_backward(buf508, relu_44, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_145
        buf510 = buf509[0]
        buf511 = buf509[1]
        del buf509
        buf512 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_49.run(buf512, relu_44, buf457, relu_47, buf484, buf510, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf457
        del buf484
        del relu_44
        del relu_47
        buf513 = buf487; del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf512, buf513, 512, 1568, grid=grid(512), stream=stream0)
        buf514 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf512, convolution_47, unsqueeze_1106, buf514, 6656, 121, grid=grid(6656), stream=stream0)
        buf515 = empty((512, ), device='cuda', dtype=torch.float32)
        buf516 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf514, squeeze_142, buf515, buf516, 512, 13, grid=grid(512), stream=stream0)
        buf517 = reinterpret_tensor(buf510, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf512, convolution_47, unsqueeze_1106, buf515, squeeze_142, buf513, primals_143, buf517, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_47
        del primals_143
        del squeeze_142
        del unsqueeze_1106
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf518 = aten.convolution_backward(buf517, cat_5, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_5
        del primals_142
        buf519 = buf518[0]
        buf520 = buf518[1]
        del buf518
        buf521 = buf515; del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_40.run(le_57, buf512, buf519, buf521, 512, 1568, grid=grid(512), stream=stream0)
        buf522 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_41.run(le_57, buf512, buf519, convolution_46, unsqueeze_1118, buf522, 6656, 121, grid=grid(6656), stream=stream0)
        buf523 = empty((512, ), device='cuda', dtype=torch.float32)
        buf525 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf522, squeeze_139, buf523, buf525, 512, 13, grid=grid(512), stream=stream0)
        buf524 = buf517; del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42.run(le_57, buf512, buf519, convolution_46, unsqueeze_1118, buf523, squeeze_139, buf521, primals_140, buf524, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del convolution_46
        del primals_140
        del squeeze_139
        del unsqueeze_1118
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf526 = aten.convolution_backward(buf524, relu_42, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_139
        buf527 = buf526[0]
        buf528 = buf526[1]
        del buf526
        buf529 = reinterpret_tensor(buf505, (256, 13), (13, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_42, buf527, buf529, 3328, 121, grid=grid(3328), stream=stream0)
        buf530 = buf506; del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf529, buf530, 256, 13, grid=grid(256), stream=stream0)
        buf531 = reinterpret_tensor(buf529, (256, 13), (1, 256), 0); del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_42, buf527, convolution_45, unsqueeze_1130, buf531, 3328, 121, grid=grid(3328), stream=stream0)
        buf532 = empty((256, ), device='cuda', dtype=torch.float32)
        buf533 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf531, squeeze_136, buf532, buf533, 256, 13, grid=grid(256), stream=stream0)
        buf534 = buf508; del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_42, buf527, convolution_45, unsqueeze_1130, buf532, squeeze_136, buf530, primals_137, buf534, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf527
        del convolution_45
        del primals_137
        del relu_42
        del squeeze_136
        del unsqueeze_1130
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf535 = aten.convolution_backward(buf534, relu_41, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_136
        buf536 = buf535[0]
        buf537 = buf535[1]
        del buf535
        buf538 = reinterpret_tensor(buf531, (256, 13), (13, 1), 0); del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_41, buf536, buf538, 3328, 121, grid=grid(3328), stream=stream0)
        buf539 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf538, buf539, 256, 13, grid=grid(256), stream=stream0)
        buf540 = reinterpret_tensor(buf538, (256, 13), (1, 256), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_41, buf536, convolution_44, unsqueeze_1142, buf540, 3328, 121, grid=grid(3328), stream=stream0)
        buf541 = empty((256, ), device='cuda', dtype=torch.float32)
        buf542 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf540, squeeze_133, buf541, buf542, 256, 13, grid=grid(256), stream=stream0)
        buf543 = buf534; del buf534  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_41, buf536, convolution_44, unsqueeze_1142, buf541, squeeze_133, buf539, primals_134, buf543, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf536
        del convolution_44
        del primals_134
        del relu_41
        del squeeze_133
        del unsqueeze_1142
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf544 = aten.convolution_backward(buf543, relu_40, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_133
        buf545 = buf544[0]
        buf546 = buf544[1]
        del buf544
        buf547 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_43.run(buf547, relu_40, buf519, le_57, buf545, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del le_57
        del relu_40
        buf548 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf547, buf548, 512, 1568, grid=grid(512), stream=stream0)
        buf549 = buf522; del buf522  # reuse
        buf574 = reinterpret_tensor(buf88, (512, 13), (13, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_59.run(buf547, convolution_43, unsqueeze_1154, convolution_40, unsqueeze_1190, buf549, buf574, 6656, 121, grid=grid(6656), stream=stream0)
        buf550 = empty((512, ), device='cuda', dtype=torch.float32)
        buf551 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf549, squeeze_130, buf550, buf551, 512, 13, grid=grid(512), stream=stream0)
        del buf549
        buf575 = empty((512, ), device='cuda', dtype=torch.float32)
        buf576 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf574, squeeze_121, buf575, buf576, 512, 13, grid=grid(512), stream=stream0)
        del buf574
        buf552 = reinterpret_tensor(buf545, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf545  # reuse
        buf577 = buf524; del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_60.run(buf547, convolution_43, unsqueeze_1154, buf550, squeeze_130, buf548, primals_131, convolution_40, unsqueeze_1190, buf575, squeeze_121, primals_122, buf552, buf577, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del buf547
        del buf550
        del buf575
        del convolution_40
        del convolution_43
        del primals_122
        del primals_131
        del squeeze_121
        del squeeze_130
        del unsqueeze_1154
        del unsqueeze_1190
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf553 = aten.convolution_backward(buf552, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf552
        del primals_130
        buf554 = buf553[0]
        buf555 = buf553[1]
        del buf553
        buf556 = reinterpret_tensor(buf540, (256, 13), (13, 1), 0); del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_39, buf554, buf556, 3328, 121, grid=grid(3328), stream=stream0)
        buf557 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf556, buf557, 256, 13, grid=grid(256), stream=stream0)
        buf558 = reinterpret_tensor(buf556, (256, 13), (1, 256), 0); del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_39, buf554, convolution_42, unsqueeze_1166, buf558, 3328, 121, grid=grid(3328), stream=stream0)
        buf559 = empty((256, ), device='cuda', dtype=torch.float32)
        buf560 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_30.run(buf558, squeeze_127, buf559, buf560, 256, 13, grid=grid(256), stream=stream0)
        del buf558
        buf561 = buf543; del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_39, buf554, convolution_42, unsqueeze_1166, buf559, squeeze_127, buf557, primals_128, buf561, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf554
        del convolution_42
        del primals_128
        del relu_39
        del squeeze_127
        del unsqueeze_1166
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf562 = aten.convolution_backward(buf561, relu_38, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf561
        del primals_127
        buf563 = buf562[0]
        buf564 = buf562[1]
        del buf562
        buf565 = empty((256, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_61.run(relu_38, buf563, buf565, 12544, 128, grid=grid(12544), stream=stream0)
        buf566 = buf559; del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_62.run(buf565, buf566, 256, 49, grid=grid(256), stream=stream0)
        buf567 = reinterpret_tensor(buf565, (256, 49), (1, 256), 0); del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_63.run(relu_38, buf563, convolution_41, unsqueeze_1178, buf567, 12544, 128, grid=grid(12544), stream=stream0)
        buf568 = empty((256, ), device='cuda', dtype=torch.float32)
        buf569 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_64.run(buf567, squeeze_124, buf568, buf569, 256, 49, grid=grid(256), stream=stream0)
        buf570 = reinterpret_tensor(buf519, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_65.run(relu_38, buf563, convolution_41, unsqueeze_1178, buf568, squeeze_124, buf566, primals_125, buf570, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del buf563
        del convolution_41
        del primals_125
        del relu_38
        del squeeze_124
        del unsqueeze_1178
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf571 = aten.convolution_backward(buf570, relu_37, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_124
        buf572 = buf571[0]
        buf573 = buf571[1]
        del buf571
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf578 = aten.convolution_backward(buf577, getitem_88, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_88
        del primals_121
        buf579 = buf578[0]
        buf580 = buf578[1]
        del buf578
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf581 = aten.max_pool2d_with_indices_backward(buf579, relu_37, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_89)
        del buf579
        buf582 = buf581
        del buf581
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf583 = aten.max_pool2d_with_indices_backward(reinterpret_tensor(buf85, (8, 256, 14, 14), (551936, 196, 14, 1), 200704), relu_37, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_89)
        del buf85
        del getitem_89
        buf584 = buf583
        del buf583
        buf585 = buf567; del buf567  # reuse
        buf587 = empty_strided((256, 49), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_66.run(relu_37, buf572, buf582, buf584, convolution_39, unsqueeze_1202, buf585, buf587, 12544, 128, grid=grid(12544), stream=stream0)
        buf586 = buf568; del buf568  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_67.run(buf585, buf586, 256, 49, grid=grid(256), stream=stream0)
        buf588 = empty((256, ), device='cuda', dtype=torch.float32)
        buf590 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_64.run(buf587, squeeze_118, buf588, buf590, 256, 49, grid=grid(256), stream=stream0)
        buf589 = buf570; del buf570  # reuse
        buf591 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_68.run(buf591, relu_37, buf572, buf582, buf584, convolution_39, unsqueeze_1202, buf588, squeeze_118, buf586, primals_119, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del convolution_39
        del primals_119
        del squeeze_118
        del unsqueeze_1202
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf592 = aten.convolution_backward(buf591, cat_4, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf591
        del cat_4
        del primals_118
        buf593 = buf592[0]
        buf594 = buf592[1]
        del buf592
        buf595 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_69.run(buf595, le_64, relu_37, buf572, buf584, buf593, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del buf572
        del le_64
        del relu_37
        buf596 = buf587; del buf587  # reuse
        buf598 = buf585; del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_70.run(buf595, convolution_38, unsqueeze_1214, buf596, buf598, 12544, 128, grid=grid(12544), stream=stream0)
        buf597 = buf588; del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_67.run(buf596, buf597, 256, 49, grid=grid(256), stream=stream0)
        buf599 = empty((256, ), device='cuda', dtype=torch.float32)
        buf600 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_64.run(buf598, squeeze_115, buf599, buf600, 256, 49, grid=grid(256), stream=stream0)
        buf601 = buf584; del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_71.run(buf595, convolution_38, unsqueeze_1214, buf599, squeeze_115, buf597, primals_116, buf601, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_38
        del primals_116
        del squeeze_115
        del unsqueeze_1214
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf602 = aten.convolution_backward(buf601, relu_35, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_115
        buf603 = buf602[0]
        buf604 = buf602[1]
        del buf602
        buf605 = empty((128, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_35, buf603, buf605, 6272, 128, grid=grid(6272), stream=stream0)
        buf606 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf605, buf606, 128, 49, grid=grid(128), stream=stream0)
        buf607 = reinterpret_tensor(buf605, (128, 49), (1, 128), 0); del buf605  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_35, buf603, convolution_37, unsqueeze_1226, buf607, 6272, 128, grid=grid(6272), stream=stream0)
        buf608 = empty((128, ), device='cuda', dtype=torch.float32)
        buf609 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf607, squeeze_112, buf608, buf609, 128, 49, grid=grid(128), stream=stream0)
        buf610 = reinterpret_tensor(buf577, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_35, buf603, convolution_37, unsqueeze_1226, buf608, squeeze_112, buf606, primals_113, buf610, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf603
        del convolution_37
        del primals_113
        del relu_35
        del squeeze_112
        del unsqueeze_1226
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf611 = aten.convolution_backward(buf610, relu_34, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_112
        buf612 = buf611[0]
        buf613 = buf611[1]
        del buf611
        buf614 = reinterpret_tensor(buf607, (128, 49), (49, 1), 0); del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_34, buf612, buf614, 6272, 128, grid=grid(6272), stream=stream0)
        buf615 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf614, buf615, 128, 49, grid=grid(128), stream=stream0)
        buf616 = reinterpret_tensor(buf614, (128, 49), (1, 128), 0); del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_34, buf612, convolution_36, unsqueeze_1238, buf616, 6272, 128, grid=grid(6272), stream=stream0)
        buf617 = empty((128, ), device='cuda', dtype=torch.float32)
        buf618 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf616, squeeze_109, buf617, buf618, 128, 49, grid=grid(128), stream=stream0)
        buf619 = buf610; del buf610  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_34, buf612, convolution_36, unsqueeze_1238, buf617, squeeze_109, buf615, primals_110, buf619, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf612
        del convolution_36
        del primals_110
        del relu_34
        del squeeze_109
        del unsqueeze_1238
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf620 = aten.convolution_backward(buf619, relu_33, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_109
        buf621 = buf620[0]
        buf622 = buf620[1]
        del buf620
        buf623 = reinterpret_tensor(buf598, (256, 49), (49, 1), 0); del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_77.run(relu_33, buf593, buf595, buf621, buf623, 12544, 128, grid=grid(12544), stream=stream0)
        buf624 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_62.run(buf623, buf624, 256, 49, grid=grid(256), stream=stream0)
        buf625 = reinterpret_tensor(buf623, (256, 49), (1, 256), 0); del buf623  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_78.run(relu_33, buf593, buf595, buf621, convolution_35, unsqueeze_1250, buf625, 12544, 128, grid=grid(12544), stream=stream0)
        buf626 = empty((256, ), device='cuda', dtype=torch.float32)
        buf628 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_64.run(buf625, squeeze_106, buf626, buf628, 256, 49, grid=grid(256), stream=stream0)
        buf627 = buf601; del buf601  # reuse
        buf629 = buf627; del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_79.run(buf629, relu_33, buf593, buf595, buf621, convolution_35, unsqueeze_1250, buf626, squeeze_106, buf624, primals_107, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del convolution_35
        del primals_107
        del squeeze_106
        del unsqueeze_1250
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf630 = aten.convolution_backward(buf629, relu_32, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf629
        del primals_106
        buf631 = buf630[0]
        buf632 = buf630[1]
        del buf630
        buf633 = reinterpret_tensor(buf616, (128, 49), (49, 1), 0); del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_32, buf631, buf633, 6272, 128, grid=grid(6272), stream=stream0)
        buf634 = buf617; del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf633, buf634, 128, 49, grid=grid(128), stream=stream0)
        buf635 = reinterpret_tensor(buf633, (128, 49), (1, 128), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_32, buf631, convolution_34, unsqueeze_1262, buf635, 6272, 128, grid=grid(6272), stream=stream0)
        buf636 = empty((128, ), device='cuda', dtype=torch.float32)
        buf637 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf635, squeeze_103, buf636, buf637, 128, 49, grid=grid(128), stream=stream0)
        buf638 = buf619; del buf619  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_32, buf631, convolution_34, unsqueeze_1262, buf636, squeeze_103, buf634, primals_104, buf638, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf631
        del convolution_34
        del primals_104
        del relu_32
        del squeeze_103
        del unsqueeze_1262
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf639 = aten.convolution_backward(buf638, relu_31, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_103
        buf640 = buf639[0]
        buf641 = buf639[1]
        del buf639
        buf642 = reinterpret_tensor(buf635, (128, 49), (49, 1), 0); del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_31, buf640, buf642, 6272, 128, grid=grid(6272), stream=stream0)
        buf643 = buf636; del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf642, buf643, 128, 49, grid=grid(128), stream=stream0)
        buf644 = reinterpret_tensor(buf642, (128, 49), (1, 128), 0); del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_31, buf640, convolution_33, unsqueeze_1274, buf644, 6272, 128, grid=grid(6272), stream=stream0)
        buf645 = empty((128, ), device='cuda', dtype=torch.float32)
        buf646 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf644, squeeze_100, buf645, buf646, 128, 49, grid=grid(128), stream=stream0)
        buf647 = buf638; del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_31, buf640, convolution_33, unsqueeze_1274, buf645, squeeze_100, buf643, primals_101, buf647, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf640
        del convolution_33
        del primals_101
        del relu_31
        del squeeze_100
        del unsqueeze_1274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf648 = aten.convolution_backward(buf647, relu_30, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_100
        buf649 = buf648[0]
        buf650 = buf648[1]
        del buf648
        buf651 = buf621; del buf621  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_80.run(buf651, relu_30, buf593, relu_33, buf595, buf649, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del buf595
        del relu_30
        del relu_33
        buf652 = buf626; del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf651, buf652, 256, 6272, grid=grid(256), stream=stream0)
        buf653 = reinterpret_tensor(buf625, (256, 49), (49, 1), 0); del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf651, convolution_32, unsqueeze_1286, buf653, 12544, 128, grid=grid(12544), stream=stream0)
        buf654 = empty((256, ), device='cuda', dtype=torch.float32)
        buf655 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf653, squeeze_97, buf654, buf655, 256, 49, grid=grid(256), stream=stream0)
        buf656 = reinterpret_tensor(buf649, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_84.run(buf651, convolution_32, unsqueeze_1286, buf654, squeeze_97, buf652, primals_98, buf656, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del convolution_32
        del primals_98
        del squeeze_97
        del unsqueeze_1286
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf657 = aten.convolution_backward(buf656, cat_3, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_3
        del primals_97
        buf658 = buf657[0]
        buf659 = buf657[1]
        del buf657
        buf660 = buf654; del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_85.run(le_71, buf651, buf658, buf660, 256, 6272, grid=grid(256), stream=stream0)
        buf661 = buf653; del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_86.run(le_71, buf651, buf658, convolution_31, unsqueeze_1298, buf661, 12544, 128, grid=grid(12544), stream=stream0)
        buf662 = empty((256, ), device='cuda', dtype=torch.float32)
        buf664 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf661, squeeze_94, buf662, buf664, 256, 49, grid=grid(256), stream=stream0)
        buf663 = buf656; del buf656  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_87.run(le_71, buf651, buf658, convolution_31, unsqueeze_1298, buf662, squeeze_94, buf660, primals_95, buf663, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del convolution_31
        del primals_95
        del squeeze_94
        del unsqueeze_1298
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf665 = aten.convolution_backward(buf663, relu_28, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf663
        del primals_94
        buf666 = buf665[0]
        buf667 = buf665[1]
        del buf665
        buf668 = reinterpret_tensor(buf644, (128, 49), (49, 1), 0); del buf644  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_28, buf666, buf668, 6272, 128, grid=grid(6272), stream=stream0)
        buf669 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf668, buf669, 128, 49, grid=grid(128), stream=stream0)
        buf670 = reinterpret_tensor(buf668, (128, 49), (1, 128), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_28, buf666, convolution_30, unsqueeze_1310, buf670, 6272, 128, grid=grid(6272), stream=stream0)
        buf671 = empty((128, ), device='cuda', dtype=torch.float32)
        buf672 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf670, squeeze_91, buf671, buf672, 128, 49, grid=grid(128), stream=stream0)
        buf673 = buf647; del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_28, buf666, convolution_30, unsqueeze_1310, buf671, squeeze_91, buf669, primals_92, buf673, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf666
        del convolution_30
        del primals_92
        del relu_28
        del squeeze_91
        del unsqueeze_1310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf674 = aten.convolution_backward(buf673, relu_27, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_91
        buf675 = buf674[0]
        buf676 = buf674[1]
        del buf674
        buf677 = reinterpret_tensor(buf670, (128, 49), (49, 1), 0); del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_27, buf675, buf677, 6272, 128, grid=grid(6272), stream=stream0)
        buf678 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf677, buf678, 128, 49, grid=grid(128), stream=stream0)
        buf679 = reinterpret_tensor(buf677, (128, 49), (1, 128), 0); del buf677  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_27, buf675, convolution_29, unsqueeze_1322, buf679, 6272, 128, grid=grid(6272), stream=stream0)
        buf680 = empty((128, ), device='cuda', dtype=torch.float32)
        buf681 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf679, squeeze_88, buf680, buf681, 128, 49, grid=grid(128), stream=stream0)
        buf682 = buf673; del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_27, buf675, convolution_29, unsqueeze_1322, buf680, squeeze_88, buf678, primals_89, buf682, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf675
        del convolution_29
        del primals_89
        del relu_27
        del squeeze_88
        del unsqueeze_1322
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf683 = aten.convolution_backward(buf682, relu_26, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_88
        buf684 = buf683[0]
        buf685 = buf683[1]
        del buf683
        buf686 = buf651; del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_88.run(buf686, relu_26, buf658, le_71, buf684, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del buf658
        del le_71
        del relu_26
        buf687 = buf662; del buf662  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf686, buf687, 256, 6272, grid=grid(256), stream=stream0)
        buf688 = buf661; del buf661  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf686, convolution_28, unsqueeze_1334, buf688, 12544, 128, grid=grid(12544), stream=stream0)
        buf689 = empty((256, ), device='cuda', dtype=torch.float32)
        buf690 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf688, squeeze_85, buf689, buf690, 256, 49, grid=grid(256), stream=stream0)
        buf691 = reinterpret_tensor(buf684, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_84.run(buf686, convolution_28, unsqueeze_1334, buf689, squeeze_85, buf687, primals_86, buf691, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del convolution_28
        del primals_86
        del squeeze_85
        del unsqueeze_1334
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf692 = aten.convolution_backward(buf691, relu_25, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_85
        buf693 = buf692[0]
        buf694 = buf692[1]
        del buf692
        buf695 = reinterpret_tensor(buf679, (128, 49), (49, 1), 0); del buf679  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_25, buf693, buf695, 6272, 128, grid=grid(6272), stream=stream0)
        buf696 = buf680; del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf695, buf696, 128, 49, grid=grid(128), stream=stream0)
        buf697 = reinterpret_tensor(buf695, (128, 49), (1, 128), 0); del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_25, buf693, convolution_27, unsqueeze_1346, buf697, 6272, 128, grid=grid(6272), stream=stream0)
        buf698 = empty((128, ), device='cuda', dtype=torch.float32)
        buf699 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf697, squeeze_82, buf698, buf699, 128, 49, grid=grid(128), stream=stream0)
        buf700 = buf682; del buf682  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_25, buf693, convolution_27, unsqueeze_1346, buf698, squeeze_82, buf696, primals_83, buf700, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf693
        del convolution_27
        del primals_83
        del relu_25
        del squeeze_82
        del unsqueeze_1346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf701 = aten.convolution_backward(buf700, relu_24, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_82
        buf702 = buf701[0]
        buf703 = buf701[1]
        del buf701
        buf704 = reinterpret_tensor(buf697, (128, 49), (49, 1), 0); del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_24, buf702, buf704, 6272, 128, grid=grid(6272), stream=stream0)
        buf705 = buf698; del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf704, buf705, 128, 49, grid=grid(128), stream=stream0)
        buf706 = reinterpret_tensor(buf704, (128, 49), (1, 128), 0); del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_24, buf702, convolution_26, unsqueeze_1358, buf706, 6272, 128, grid=grid(6272), stream=stream0)
        buf707 = empty((128, ), device='cuda', dtype=torch.float32)
        buf708 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf706, squeeze_79, buf707, buf708, 128, 49, grid=grid(128), stream=stream0)
        buf709 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_24, buf702, convolution_26, unsqueeze_1358, buf707, squeeze_79, buf705, primals_80, buf709, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf702
        del convolution_26
        del primals_80
        del relu_24
        del squeeze_79
        del unsqueeze_1358
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf710 = aten.convolution_backward(buf709, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_79
        buf711 = buf710[0]
        buf712 = buf710[1]
        del buf710
        buf713 = buf689; del buf689  # reuse
        buf714 = empty((256, ), device='cuda', dtype=torch.float32)
        buf716 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_89.run(relu_23, buf593, buf686, buf711, convolution_25, unsqueeze_1370, squeeze_76, buf713, buf714, buf716, 256, 6272, grid=grid(256), stream=stream0)
        buf717 = buf691; del buf691  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(relu_23, buf593, buf686, buf711, convolution_25, unsqueeze_1370, buf714, squeeze_76, buf713, primals_77, buf717, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del convolution_25
        del primals_77
        del squeeze_76
        del unsqueeze_1370
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf718 = aten.convolution_backward(buf717, cat_2, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf717
        del cat_2
        del primals_76
        buf719 = buf718[0]
        buf720 = buf718[1]
        del buf718
        buf721 = buf686; del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_91.run(buf721, le_78, relu_23, buf593, buf711, buf719, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del le_78
        del relu_23
        buf722 = buf714; del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf721, buf722, 256, 6272, grid=grid(256), stream=stream0)
        buf723 = buf688; del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf721, convolution_24, unsqueeze_1382, buf723, 12544, 128, grid=grid(12544), stream=stream0)
        buf724 = empty((256, ), device='cuda', dtype=torch.float32)
        buf725 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf723, squeeze_73, buf724, buf725, 256, 49, grid=grid(256), stream=stream0)
        buf726 = reinterpret_tensor(buf711, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_84.run(buf721, convolution_24, unsqueeze_1382, buf724, squeeze_73, buf722, primals_74, buf726, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del convolution_24
        del primals_74
        del squeeze_73
        del unsqueeze_1382
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf727 = aten.convolution_backward(buf726, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_73
        buf728 = buf727[0]
        buf729 = buf727[1]
        del buf727
        buf730 = reinterpret_tensor(buf706, (128, 49), (49, 1), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_21, buf728, buf730, 6272, 128, grid=grid(6272), stream=stream0)
        buf731 = buf707; del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf730, buf731, 128, 49, grid=grid(128), stream=stream0)
        buf732 = reinterpret_tensor(buf730, (128, 49), (1, 128), 0); del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_21, buf728, convolution_23, unsqueeze_1394, buf732, 6272, 128, grid=grid(6272), stream=stream0)
        buf733 = empty((128, ), device='cuda', dtype=torch.float32)
        buf734 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf732, squeeze_70, buf733, buf734, 128, 49, grid=grid(128), stream=stream0)
        buf735 = buf709; del buf709  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_21, buf728, convolution_23, unsqueeze_1394, buf733, squeeze_70, buf731, primals_71, buf735, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf728
        del convolution_23
        del primals_71
        del relu_21
        del squeeze_70
        del unsqueeze_1394
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf736 = aten.convolution_backward(buf735, relu_20, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_70
        buf737 = buf736[0]
        buf738 = buf736[1]
        del buf736
        buf739 = reinterpret_tensor(buf732, (128, 49), (49, 1), 0); del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_20, buf737, buf739, 6272, 128, grid=grid(6272), stream=stream0)
        buf740 = buf733; del buf733  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf739, buf740, 128, 49, grid=grid(128), stream=stream0)
        buf741 = reinterpret_tensor(buf739, (128, 49), (1, 128), 0); del buf739  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_20, buf737, convolution_22, unsqueeze_1406, buf741, 6272, 128, grid=grid(6272), stream=stream0)
        buf742 = empty((128, ), device='cuda', dtype=torch.float32)
        buf743 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf741, squeeze_67, buf742, buf743, 128, 49, grid=grid(128), stream=stream0)
        buf744 = buf735; del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_20, buf737, convolution_22, unsqueeze_1406, buf742, squeeze_67, buf740, primals_68, buf744, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf737
        del convolution_22
        del primals_68
        del relu_20
        del squeeze_67
        del unsqueeze_1406
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf745 = aten.convolution_backward(buf744, relu_19, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_67
        buf746 = buf745[0]
        buf747 = buf745[1]
        del buf745
        buf748 = buf724; del buf724  # reuse
        buf749 = empty((256, ), device='cuda', dtype=torch.float32)
        buf751 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_92.run(relu_19, buf719, buf721, buf746, convolution_21, unsqueeze_1418, squeeze_64, buf748, buf749, buf751, 256, 6272, grid=grid(256), stream=stream0)
        buf752 = buf726; del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_93.run(relu_19, buf719, buf721, buf746, convolution_21, unsqueeze_1418, buf749, squeeze_64, buf748, primals_65, buf752, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del convolution_21
        del primals_65
        del squeeze_64
        del unsqueeze_1418
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf753 = aten.convolution_backward(buf752, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf752
        del primals_64
        buf754 = buf753[0]
        buf755 = buf753[1]
        del buf753
        buf756 = reinterpret_tensor(buf741, (128, 49), (49, 1), 0); del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_18, buf754, buf756, 6272, 128, grid=grid(6272), stream=stream0)
        buf757 = buf742; del buf742  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf756, buf757, 128, 49, grid=grid(128), stream=stream0)
        buf758 = reinterpret_tensor(buf756, (128, 49), (1, 128), 0); del buf756  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_18, buf754, convolution_20, unsqueeze_1430, buf758, 6272, 128, grid=grid(6272), stream=stream0)
        buf759 = empty((128, ), device='cuda', dtype=torch.float32)
        buf760 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf758, squeeze_61, buf759, buf760, 128, 49, grid=grid(128), stream=stream0)
        buf761 = buf744; del buf744  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_18, buf754, convolution_20, unsqueeze_1430, buf759, squeeze_61, buf757, primals_62, buf761, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf754
        del convolution_20
        del primals_62
        del relu_18
        del squeeze_61
        del unsqueeze_1430
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf762 = aten.convolution_backward(buf761, relu_17, primals_61, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_61
        buf763 = buf762[0]
        buf764 = buf762[1]
        del buf762
        buf765 = reinterpret_tensor(buf758, (128, 49), (49, 1), 0); del buf758  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_17, buf763, buf765, 6272, 128, grid=grid(6272), stream=stream0)
        buf766 = buf759; del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf765, buf766, 128, 49, grid=grid(128), stream=stream0)
        buf767 = reinterpret_tensor(buf765, (128, 49), (1, 128), 0); del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_17, buf763, convolution_19, unsqueeze_1442, buf767, 6272, 128, grid=grid(6272), stream=stream0)
        buf768 = empty((128, ), device='cuda', dtype=torch.float32)
        buf769 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf767, squeeze_58, buf768, buf769, 128, 49, grid=grid(128), stream=stream0)
        buf770 = buf761; del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_17, buf763, convolution_19, unsqueeze_1442, buf768, squeeze_58, buf766, primals_59, buf770, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf763
        del convolution_19
        del primals_59
        del relu_17
        del squeeze_58
        del unsqueeze_1442
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf771 = aten.convolution_backward(buf770, relu_16, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_58
        buf772 = buf771[0]
        buf773 = buf771[1]
        del buf771
        buf774 = buf721; del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_94.run(buf774, relu_16, buf719, relu_19, buf746, buf772, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del buf719
        del buf746
        del relu_16
        del relu_19
        buf775 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf774, buf775, 256, 6272, grid=grid(256), stream=stream0)
        buf776 = buf723; del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf774, convolution_18, unsqueeze_1454, buf776, 12544, 128, grid=grid(12544), stream=stream0)
        buf777 = empty((256, ), device='cuda', dtype=torch.float32)
        buf778 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf776, squeeze_55, buf777, buf778, 256, 49, grid=grid(256), stream=stream0)
        buf779 = reinterpret_tensor(buf772, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_84.run(buf774, convolution_18, unsqueeze_1454, buf777, squeeze_55, buf775, primals_56, buf779, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del convolution_18
        del primals_56
        del squeeze_55
        del unsqueeze_1454
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf780 = aten.convolution_backward(buf779, cat_1, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_1
        del primals_55
        buf781 = buf780[0]
        buf782 = buf780[1]
        del buf780
        buf783 = buf777; del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_85.run(le_85, buf774, buf781, buf783, 256, 6272, grid=grid(256), stream=stream0)
        buf784 = buf776; del buf776  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_86.run(le_85, buf774, buf781, convolution_17, unsqueeze_1466, buf784, 12544, 128, grid=grid(12544), stream=stream0)
        buf785 = empty((256, ), device='cuda', dtype=torch.float32)
        buf787 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf784, squeeze_52, buf785, buf787, 256, 49, grid=grid(256), stream=stream0)
        buf786 = buf779; del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_87.run(le_85, buf774, buf781, convolution_17, unsqueeze_1466, buf785, squeeze_52, buf783, primals_53, buf786, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del convolution_17
        del primals_53
        del squeeze_52
        del unsqueeze_1466
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf788 = aten.convolution_backward(buf786, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_52
        buf789 = buf788[0]
        buf790 = buf788[1]
        del buf788
        buf791 = reinterpret_tensor(buf767, (128, 49), (49, 1), 0); del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_14, buf789, buf791, 6272, 128, grid=grid(6272), stream=stream0)
        buf792 = buf768; del buf768  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf791, buf792, 128, 49, grid=grid(128), stream=stream0)
        buf793 = reinterpret_tensor(buf791, (128, 49), (1, 128), 0); del buf791  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_14, buf789, convolution_16, unsqueeze_1478, buf793, 6272, 128, grid=grid(6272), stream=stream0)
        buf794 = empty((128, ), device='cuda', dtype=torch.float32)
        buf795 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf793, squeeze_49, buf794, buf795, 128, 49, grid=grid(128), stream=stream0)
        buf796 = buf770; del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_14, buf789, convolution_16, unsqueeze_1478, buf794, squeeze_49, buf792, primals_50, buf796, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf789
        del convolution_16
        del primals_50
        del relu_14
        del squeeze_49
        del unsqueeze_1478
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf797 = aten.convolution_backward(buf796, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_49
        buf798 = buf797[0]
        buf799 = buf797[1]
        del buf797
        buf800 = reinterpret_tensor(buf793, (128, 49), (49, 1), 0); del buf793  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_13, buf798, buf800, 6272, 128, grid=grid(6272), stream=stream0)
        buf801 = buf794; del buf794  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf800, buf801, 128, 49, grid=grid(128), stream=stream0)
        buf802 = reinterpret_tensor(buf800, (128, 49), (1, 128), 0); del buf800  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_13, buf798, convolution_15, unsqueeze_1490, buf802, 6272, 128, grid=grid(6272), stream=stream0)
        buf803 = empty((128, ), device='cuda', dtype=torch.float32)
        buf804 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf802, squeeze_46, buf803, buf804, 128, 49, grid=grid(128), stream=stream0)
        buf805 = buf796; del buf796  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_13, buf798, convolution_15, unsqueeze_1490, buf803, squeeze_46, buf801, primals_47, buf805, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf798
        del convolution_15
        del primals_47
        del relu_13
        del squeeze_46
        del unsqueeze_1490
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf806 = aten.convolution_backward(buf805, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_46
        buf807 = buf806[0]
        buf808 = buf806[1]
        del buf806
        buf809 = buf774; del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_88.run(buf809, relu_12, buf781, le_85, buf807, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del le_85
        del relu_12
        buf810 = buf785; del buf785  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf809, buf810, 256, 6272, grid=grid(256), stream=stream0)
        buf811 = buf784; del buf784  # reuse
        buf836 = reinterpret_tensor(buf596, (256, 49), (49, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_95.run(buf809, convolution_14, unsqueeze_1502, convolution_11, unsqueeze_1538, buf811, buf836, 12544, 128, grid=grid(12544), stream=stream0)
        buf812 = empty((256, ), device='cuda', dtype=torch.float32)
        buf813 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf811, squeeze_43, buf812, buf813, 256, 49, grid=grid(256), stream=stream0)
        del buf811
        buf837 = empty((256, ), device='cuda', dtype=torch.float32)
        buf838 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf836, squeeze_34, buf837, buf838, 256, 49, grid=grid(256), stream=stream0)
        buf814 = reinterpret_tensor(buf807, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf807  # reuse
        buf839 = buf786; del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_96.run(buf809, convolution_14, unsqueeze_1502, buf812, squeeze_43, buf810, primals_44, convolution_11, unsqueeze_1538, buf837, squeeze_34, primals_35, buf814, buf839, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del buf809
        del buf812
        del buf837
        del convolution_11
        del convolution_14
        del primals_35
        del primals_44
        del squeeze_34
        del squeeze_43
        del unsqueeze_1502
        del unsqueeze_1538
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf815 = aten.convolution_backward(buf814, relu_11, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf814
        del primals_43
        buf816 = buf815[0]
        buf817 = buf815[1]
        del buf815
        buf818 = reinterpret_tensor(buf802, (128, 49), (49, 1), 0); del buf802  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(relu_11, buf816, buf818, 6272, 128, grid=grid(6272), stream=stream0)
        buf819 = buf803; del buf803  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_73.run(buf818, buf819, 128, 49, grid=grid(128), stream=stream0)
        buf820 = reinterpret_tensor(buf818, (128, 49), (1, 128), 0); del buf818  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_11, buf816, convolution_13, unsqueeze_1514, buf820, 6272, 128, grid=grid(6272), stream=stream0)
        buf821 = empty((128, ), device='cuda', dtype=torch.float32)
        buf822 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf820, squeeze_40, buf821, buf822, 128, 49, grid=grid(128), stream=stream0)
        del buf820
        buf823 = buf805; del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76.run(relu_11, buf816, convolution_13, unsqueeze_1514, buf821, squeeze_40, buf819, primals_41, buf823, 6272, 128, grid=grid(6272, 128), stream=stream0)
        del buf816
        del convolution_13
        del primals_41
        del relu_11
        del squeeze_40
        del unsqueeze_1514
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf824 = aten.convolution_backward(buf823, relu_10, primals_40, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf823
        del primals_40
        buf825 = buf824[0]
        buf826 = buf824[1]
        del buf824
        buf827 = empty((128, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_97.run(relu_10, buf825, buf827, 25088, 128, grid=grid(25088), stream=stream0)
        buf828 = buf821; del buf821  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_98.run(buf827, buf828, 128, 196, grid=grid(128), stream=stream0)
        buf829 = reinterpret_tensor(buf827, (128, 196), (1, 128), 0); del buf827  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_99.run(relu_10, buf825, convolution_12, unsqueeze_1526, buf829, 25088, 128, grid=grid(25088), stream=stream0)
        buf830 = empty((128, ), device='cuda', dtype=torch.float32)
        buf831 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(buf829, squeeze_37, buf830, buf831, 128, 196, grid=grid(128), stream=stream0)
        buf832 = reinterpret_tensor(buf781, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_101.run(relu_10, buf825, convolution_12, unsqueeze_1526, buf830, squeeze_37, buf828, primals_38, buf832, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del buf825
        del convolution_12
        del primals_38
        del relu_10
        del squeeze_37
        del unsqueeze_1526
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf833 = aten.convolution_backward(buf832, relu_9, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_37
        buf834 = buf833[0]
        buf835 = buf833[1]
        del buf833
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf840 = aten.convolution_backward(buf839, getitem_24, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_24
        del primals_34
        buf841 = buf840[0]
        buf842 = buf840[1]
        del buf840
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf843 = aten.max_pool2d_with_indices_backward(buf841, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_25)
        del buf841
        buf844 = buf843
        del buf843
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf845 = aten.max_pool2d_with_indices_backward(reinterpret_tensor(buf593, (8, 128, 28, 28), (903168, 784, 28, 1), 401408), relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_25)
        del buf593
        del getitem_25
        buf846 = buf845
        del buf845
        buf847 = buf829; del buf829  # reuse
        buf849 = empty_strided((128, 196), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_102.run(relu_9, buf834, buf844, buf846, convolution_10, unsqueeze_1550, buf847, buf849, 25088, 128, grid=grid(25088), stream=stream0)
        buf848 = buf830; del buf830  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_103.run(buf847, buf848, 128, 196, grid=grid(128), stream=stream0)
        buf850 = empty((128, ), device='cuda', dtype=torch.float32)
        buf852 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(buf849, squeeze_31, buf850, buf852, 128, 196, grid=grid(128), stream=stream0)
        buf851 = buf832; del buf832  # reuse
        buf853 = buf851; del buf851  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_104.run(buf853, relu_9, buf834, buf844, buf846, convolution_10, unsqueeze_1550, buf850, squeeze_31, buf848, primals_32, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del convolution_10
        del primals_32
        del squeeze_31
        del unsqueeze_1550
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf854 = aten.convolution_backward(buf853, cat, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf853
        del cat
        del primals_31
        buf855 = buf854[0]
        buf856 = buf854[1]
        del buf854
        buf857 = buf844; del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_105.run(buf857, le_92, relu_9, buf834, buf846, buf855, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del le_92
        del relu_9
        buf858 = buf849; del buf849  # reuse
        buf860 = buf847; del buf847  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_106.run(buf857, convolution_9, unsqueeze_1562, buf858, buf860, 25088, 128, grid=grid(25088), stream=stream0)
        buf859 = buf850; del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_103.run(buf858, buf859, 128, 196, grid=grid(128), stream=stream0)
        buf861 = empty((128, ), device='cuda', dtype=torch.float32)
        buf862 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(buf860, squeeze_28, buf861, buf862, 128, 196, grid=grid(128), stream=stream0)
        buf863 = buf846; del buf846  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_107.run(buf857, convolution_9, unsqueeze_1562, buf861, squeeze_28, buf859, primals_29, buf863, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_9
        del primals_29
        del squeeze_28
        del unsqueeze_1562
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf864 = aten.convolution_backward(buf863, relu_7, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_28
        buf865 = buf864[0]
        buf866 = buf864[1]
        del buf864
        buf867 = reinterpret_tensor(buf836, (64, 196), (196, 1), 0); del buf836  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_108.run(relu_7, buf865, buf867, 12544, 128, grid=grid(12544), stream=stream0)
        buf868 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_109.run(buf867, buf868, 64, 196, grid=grid(64), stream=stream0)
        buf869 = reinterpret_tensor(buf867, (64, 196), (1, 64), 0); del buf867  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_110.run(relu_7, buf865, convolution_8, unsqueeze_1574, buf869, 12544, 128, grid=grid(12544), stream=stream0)
        buf870 = empty((64, ), device='cuda', dtype=torch.float32)
        buf871 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_111.run(buf869, squeeze_25, buf870, buf871, 64, 196, grid=grid(64), stream=stream0)
        buf872 = reinterpret_tensor(buf839, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf839  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_112.run(relu_7, buf865, convolution_8, unsqueeze_1574, buf870, squeeze_25, buf868, primals_26, buf872, 25088, 64, grid=grid(25088, 64), stream=stream0)
        del buf865
        del convolution_8
        del primals_26
        del relu_7
        del squeeze_25
        del unsqueeze_1574
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf873 = aten.convolution_backward(buf872, relu_6, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_25
        buf874 = buf873[0]
        buf875 = buf873[1]
        del buf873
        buf876 = reinterpret_tensor(buf869, (64, 196), (196, 1), 0); del buf869  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_108.run(relu_6, buf874, buf876, 12544, 128, grid=grid(12544), stream=stream0)
        buf877 = buf870; del buf870  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_109.run(buf876, buf877, 64, 196, grid=grid(64), stream=stream0)
        buf878 = reinterpret_tensor(buf876, (64, 196), (1, 64), 0); del buf876  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_110.run(relu_6, buf874, convolution_7, unsqueeze_1586, buf878, 12544, 128, grid=grid(12544), stream=stream0)
        buf879 = empty((64, ), device='cuda', dtype=torch.float32)
        buf880 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_111.run(buf878, squeeze_22, buf879, buf880, 64, 196, grid=grid(64), stream=stream0)
        buf881 = buf872; del buf872  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_112.run(relu_6, buf874, convolution_7, unsqueeze_1586, buf879, squeeze_22, buf877, primals_23, buf881, 25088, 64, grid=grid(25088, 64), stream=stream0)
        del buf874
        del convolution_7
        del primals_23
        del relu_6
        del squeeze_22
        del unsqueeze_1586
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf882 = aten.convolution_backward(buf881, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_22
        buf883 = buf882[0]
        buf884 = buf882[1]
        del buf882
        buf885 = reinterpret_tensor(buf860, (128, 196), (196, 1), 0); del buf860  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_113.run(relu_5, buf855, buf857, buf883, buf885, 25088, 128, grid=grid(25088), stream=stream0)
        buf886 = buf861; del buf861  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_98.run(buf885, buf886, 128, 196, grid=grid(128), stream=stream0)
        buf887 = reinterpret_tensor(buf885, (128, 196), (1, 128), 0); del buf885  # reuse
        buf913 = buf858; del buf858  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_114.run(relu_5, buf855, buf857, buf883, convolution_6, unsqueeze_1598, convolution_3, unsqueeze_1634, buf887, buf913, 25088, 128, grid=grid(25088), stream=stream0)
        buf888 = empty((128, ), device='cuda', dtype=torch.float32)
        buf890 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(buf887, squeeze_19, buf888, buf890, 128, 196, grid=grid(128), stream=stream0)
        buf914 = empty((128, ), device='cuda', dtype=torch.float32)
        buf916 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(buf913, squeeze_10, buf914, buf916, 128, 196, grid=grid(128), stream=stream0)
        buf889 = buf863; del buf863  # reuse
        buf915 = reinterpret_tensor(buf834, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf834  # reuse
        buf891 = buf889; del buf889  # reuse
        buf917 = buf915; del buf915  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_115.run(buf891, buf917, relu_5, buf855, buf857, buf883, convolution_6, unsqueeze_1598, buf888, squeeze_19, buf886, convolution_3, unsqueeze_1634, buf914, squeeze_10, primals_20, primals_11, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del buf857
        del buf883
        del buf888
        del buf914
        del convolution_3
        del convolution_6
        del primals_11
        del primals_20
        del relu_5
        del squeeze_10
        del squeeze_19
        del unsqueeze_1598
        del unsqueeze_1634
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf892 = aten.convolution_backward(buf891, relu_4, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf891
        del primals_19
        buf893 = buf892[0]
        buf894 = buf892[1]
        del buf892
        buf895 = reinterpret_tensor(buf878, (64, 196), (196, 1), 0); del buf878  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_108.run(relu_4, buf893, buf895, 12544, 128, grid=grid(12544), stream=stream0)
        buf896 = buf879; del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_109.run(buf895, buf896, 64, 196, grid=grid(64), stream=stream0)
        buf897 = reinterpret_tensor(buf895, (64, 196), (1, 64), 0); del buf895  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_110.run(relu_4, buf893, convolution_5, unsqueeze_1610, buf897, 12544, 128, grid=grid(12544), stream=stream0)
        buf898 = empty((64, ), device='cuda', dtype=torch.float32)
        buf899 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_111.run(buf897, squeeze_16, buf898, buf899, 64, 196, grid=grid(64), stream=stream0)
        del buf897
        buf900 = buf881; del buf881  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_112.run(relu_4, buf893, convolution_5, unsqueeze_1610, buf898, squeeze_16, buf896, primals_17, buf900, 25088, 64, grid=grid(25088, 64), stream=stream0)
        del buf893
        del convolution_5
        del primals_17
        del relu_4
        del squeeze_16
        del unsqueeze_1610
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf901 = aten.convolution_backward(buf900, relu_3, primals_16, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf900
        del primals_16
        buf902 = buf901[0]
        buf903 = buf901[1]
        del buf901
        buf904 = empty((64, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_116.run(relu_3, buf902, buf904, 50176, 128, grid=grid(50176), stream=stream0)
        buf905 = buf898; del buf898  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_117.run(buf904, buf905, 64, 784, grid=grid(64), stream=stream0)
        buf906 = reinterpret_tensor(buf904, (64, 784), (1, 64), 0); del buf904  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_118.run(relu_3, buf902, convolution_4, unsqueeze_1622, buf906, 50176, 128, grid=grid(50176), stream=stream0)
        buf907 = empty((64, ), device='cuda', dtype=torch.float32)
        buf908 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_119.run(buf906, squeeze_13, buf907, buf908, 64, 784, grid=grid(64), stream=stream0)
        buf909 = reinterpret_tensor(buf855, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf855  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_120.run(relu_3, buf902, convolution_4, unsqueeze_1622, buf907, squeeze_13, buf905, primals_14, buf909, 100352, 64, grid=grid(100352, 64), stream=stream0)
        del buf902
        del buf907
        del convolution_4
        del primals_14
        del relu_3
        del squeeze_13
        del unsqueeze_1622
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf910 = aten.convolution_backward(buf909, relu_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_13
        buf911 = buf910[0]
        buf912 = buf910[1]
        del buf910
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf918 = aten.convolution_backward(buf917, getitem_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf917
        del getitem_6
        del primals_10
        buf919 = buf918[0]
        buf920 = buf918[1]
        del buf918
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf921 = aten.max_pool2d_with_indices_backward(buf919, relu_2, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_7)
        del buf919
        del getitem_7
        buf922 = buf921
        del buf921
        buf923 = reinterpret_tensor(buf913, (32, 784), (1, 32), 0); del buf913  # reuse
        buf925 = reinterpret_tensor(buf887, (32, 784), (1, 32), 0); del buf887  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_121.run(relu_2, buf911, buf922, convolution_2, unsqueeze_1646, buf923, buf925, 25088, 128, grid=grid(25088), stream=stream0)
        buf924 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_122.run(buf923, buf924, 32, 784, grid=grid(32), stream=stream0)
        del buf923
        buf926 = empty((32, ), device='cuda', dtype=torch.float32)
        buf928 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_123.run(buf925, squeeze_7, buf926, buf928, 32, 784, grid=grid(32), stream=stream0)
        del buf925
        buf927 = buf922; del buf922  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_124.run(buf927, relu_2, buf911, convolution_2, unsqueeze_1646, buf926, squeeze_7, buf924, primals_8, 100352, 32, grid=grid(100352, 32), stream=stream0)
        del buf911
        del buf926
        del convolution_2
        del primals_8
        del relu_2
        del squeeze_7
        del unsqueeze_1646
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf929 = aten.convolution_backward(buf927, relu_1, primals_7, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf927
        del primals_7
        buf930 = buf929[0]
        buf931 = buf929[1]
        del buf929
        buf932 = reinterpret_tensor(buf906, (16, 3136), (3136, 1), 0); del buf906  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_125.run(relu_1, buf930, buf932, 50176, 128, grid=grid(50176), stream=stream0)
        buf933 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_126.run(buf932, buf933, 16, 3136, grid=grid(16), stream=stream0)
        buf934 = reinterpret_tensor(buf932, (16, 3136), (1, 16), 0); del buf932  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_127.run(relu_1, buf930, convolution_1, unsqueeze_1658, buf934, 50176, 128, grid=grid(50176), stream=stream0)
        buf935 = empty((16, ), device='cuda', dtype=torch.float32)
        buf936 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_128.run(buf934, squeeze_4, buf935, buf936, 16, 3136, grid=grid(16), stream=stream0)
        buf937 = reinterpret_tensor(buf909, (8, 16, 224, 224), (802816, 1, 3584, 16), 0); del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_129.run(relu_1, buf930, convolution_1, unsqueeze_1658, buf935, squeeze_4, buf933, primals_5, buf937, 401408, 16, grid=grid(401408, 16), stream=stream0)
        del buf930
        del convolution_1
        del primals_5
        del relu_1
        del squeeze_4
        del unsqueeze_1658
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf938 = aten.convolution_backward(buf937, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_4
        buf939 = buf938[0]
        buf940 = buf938[1]
        del buf938
        buf941 = reinterpret_tensor(buf934, (16, 3136), (3136, 1), 0); del buf934  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_125.run(relu, buf939, buf941, 50176, 128, grid=grid(50176), stream=stream0)
        buf942 = buf935; del buf935  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_126.run(buf941, buf942, 16, 3136, grid=grid(16), stream=stream0)
        buf943 = reinterpret_tensor(buf941, (16, 3136), (1, 16), 0); del buf941  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_127.run(relu, buf939, convolution, unsqueeze_1670, buf943, 50176, 128, grid=grid(50176), stream=stream0)
        buf944 = empty((16, ), device='cuda', dtype=torch.float32)
        buf945 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_128.run(buf943, squeeze_1, buf944, buf945, 16, 3136, grid=grid(16), stream=stream0)
        del buf943
        buf946 = buf937; del buf937  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_129.run(relu, buf939, convolution, unsqueeze_1670, buf944, squeeze_1, buf942, primals_2, buf946, 401408, 16, grid=grid(401408, 16), stream=stream0)
        del buf939
        del buf944
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_1670
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf947 = aten.convolution_backward(buf946, primals_633, primals_1, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf946
        del primals_1
        del primals_633
        buf948 = buf947[1]
        return (buf948, buf945, buf942, buf940, buf936, buf933, buf931, buf928, buf924, buf920, buf916, buf886, buf912, buf908, buf905, buf903, buf899, buf896, buf894, buf890, buf886, buf884, buf880, buf877, buf875, buf871, buf868, buf866, buf862, buf859, buf856, buf852, buf848, buf842, buf838, buf810, buf835, buf831, buf828, buf826, buf822, buf819, buf817, buf813, buf810, buf808, buf804, buf801, buf799, buf795, buf792, buf790, buf787, buf783, buf782, buf778, buf775, buf773, buf769, buf766, buf764, buf760, buf757, buf755, buf751, buf748, buf747, buf743, buf740, buf738, buf734, buf731, buf729, buf725, buf722, buf720, buf716, buf713, buf712, buf708, buf705, buf703, buf699, buf696, buf694, buf690, buf687, buf685, buf681, buf678, buf676, buf672, buf669, buf667, buf664, buf660, buf659, buf655, buf652, buf650, buf646, buf643, buf641, buf637, buf634, buf632, buf628, buf624, buf622, buf618, buf615, buf613, buf609, buf606, buf604, buf600, buf597, buf594, buf590, buf586, buf580, buf576, buf548, buf573, buf569, buf566, buf564, buf560, buf557, buf555, buf551, buf548, buf546, buf542, buf539, buf537, buf533, buf530, buf528, buf525, buf521, buf520, buf516, buf513, buf511, buf507, buf504, buf502, buf498, buf495, buf493, buf489, buf486, buf485, buf481, buf478, buf476, buf472, buf469, buf467, buf463, buf460, buf458, buf454, buf451, buf450, buf446, buf443, buf441, buf437, buf434, buf432, buf428, buf425, buf423, buf419, buf416, buf414, buf410, buf407, buf405, buf402, buf398, buf397, buf393, buf390, buf388, buf384, buf381, buf379, buf375, buf372, buf370, buf366, buf363, buf362, buf358, buf355, buf353, buf349, buf346, buf344, buf340, buf337, buf335, buf331, buf328, buf327, buf323, buf320, buf318, buf314, buf311, buf309, buf305, buf302, buf300, buf296, buf293, buf291, buf287, buf284, buf282, buf279, buf275, buf274, buf270, buf267, buf265, buf261, buf258, buf256, buf252, buf249, buf247, buf243, buf240, buf239, buf235, buf232, buf230, buf226, buf223, buf221, buf217, buf214, buf212, buf208, buf205, buf204, buf200, buf197, buf195, buf191, buf188, buf186, buf182, buf179, buf177, buf173, buf170, buf168, buf164, buf161, buf159, buf156, buf152, buf151, buf147, buf144, buf142, buf138, buf135, buf133, buf129, buf126, buf124, buf120, buf116, buf114, buf110, buf107, buf105, buf101, buf98, buf96, buf92, buf89, buf86, buf83, buf79, buf74, buf70, buf42, buf67, buf63, buf60, buf58, buf54, buf51, buf49, buf45, buf42, buf40, buf36, buf33, buf31, buf27, buf24, buf22, buf18, buf14, buf12, buf8, buf5, buf3, buf0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 16, 3, 3), (144, 1, 48, 16), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((512, 2816, 1, 1), (2816, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1024, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1000, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.int64)
    convolution_3 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    getitem_24 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.int64)
    convolution_11 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 1152, 28, 28), (903168, 1, 32256, 1152), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    getitem_88 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    getitem_89 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.int64)
    convolution_40 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_48 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_49 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_51 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_52 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_53 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_54 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_55 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_56 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_58 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_59 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_60 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_61 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_196 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_62 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_199 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_63 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_202 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_8 = rand_strided((8, 2048, 14, 14), (401408, 1, 28672, 2048), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_205 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_65 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_208 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_66 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_67 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_68 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_217 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_69 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_220 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_70 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_9 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_226 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_72 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_73 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_232 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_74 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_235 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_75 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_238 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_76 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_241 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_77 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_81 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_244 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_10 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_247 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_79 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_80 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_84 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_253 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_81 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_85 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_82 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_86 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_259 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_83 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_87 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_262 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_84 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_265 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_11 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_89 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_86 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_90 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_271 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_87 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_91 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_274 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_88 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_92 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_277 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_89 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_93 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_280 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_90 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_94 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_91 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_95 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_12 = rand_strided((8, 2816, 14, 14), (551936, 1, 39424, 2816), device='cuda:0', dtype=torch.float32)
    convolution_96 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_289 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_93 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    getitem_210 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    getitem_211 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.int64)
    convolution_97 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_292 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_98 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_295 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_94 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_99 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_298 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_95 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_100 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_301 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_96 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    convolution_101 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_304 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_97 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_102 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_307 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_98 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_103 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_310 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_13 = rand_strided((8, 2560, 7, 7), (125440, 1, 17920, 2560), device='cuda:0', dtype=torch.float32)
    convolution_104 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_313 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.bool)
    unsqueeze_422 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.bool)
    unsqueeze_434 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_446 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_470 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_494 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_518 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_530 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_542 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_554 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_566 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_578 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_590 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_602 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_15 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_614 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_626 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_638 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_650 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_662 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_674 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_686 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_22 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_698 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_710 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_722 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_734 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_746 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_758 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_770 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_29 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_782 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_794 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_806 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_818 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_830 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_842 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_854 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_36 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_866 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_878 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_890 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_902 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_914 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_926 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_938 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_43 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_950 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_962 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_974 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_986 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_998 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1010 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1022 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_50 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_1034 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1046 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1058 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1070 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1082 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1094 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1106 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_57 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_1118 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1130 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1142 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1154 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1166 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1178 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1190 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1202 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_64 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_1214 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1226 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1238 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1250 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1262 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1274 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1286 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_71 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_1298 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1310 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1322 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1334 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1346 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1358 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1370 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_78 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_1382 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1394 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1406 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1418 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1430 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1442 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1454 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_85 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_1466 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1478 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1490 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1502 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1514 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1526 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1538 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1550 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_92 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_1562 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1574 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1586 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1598 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1610 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1622 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1634 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1646 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1658 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1670 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, primals_314, primals_316, primals_633, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, cat, convolution_10, squeeze_31, relu_9, getitem_24, getitem_25, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_10, convolution_13, squeeze_40, relu_11, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, cat_1, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, cat_2, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, relu_24, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, relu_26, convolution_29, squeeze_88, relu_27, convolution_30, squeeze_91, relu_28, convolution_31, squeeze_94, cat_3, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, relu_31, convolution_34, squeeze_103, relu_32, convolution_35, squeeze_106, relu_33, convolution_36, squeeze_109, relu_34, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, cat_4, convolution_39, squeeze_118, relu_37, getitem_88, getitem_89, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, relu_42, convolution_46, squeeze_139, cat_5, convolution_47, squeeze_142, relu_44, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, relu_46, convolution_50, squeeze_151, relu_47, convolution_51, squeeze_154, relu_48, convolution_52, squeeze_157, relu_49, convolution_53, squeeze_160, cat_6, convolution_54, squeeze_163, relu_51, convolution_55, squeeze_166, relu_52, convolution_56, squeeze_169, relu_53, convolution_57, squeeze_172, relu_54, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, relu_56, convolution_60, squeeze_181, cat_7, convolution_61, squeeze_184, relu_58, convolution_62, squeeze_187, relu_59, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, relu_61, convolution_65, squeeze_196, relu_62, convolution_66, squeeze_199, relu_63, convolution_67, squeeze_202, cat_8, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, relu_66, convolution_70, squeeze_211, relu_67, convolution_71, squeeze_214, relu_68, convolution_72, squeeze_217, relu_69, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, cat_9, convolution_75, squeeze_226, relu_72, convolution_76, squeeze_229, relu_73, convolution_77, squeeze_232, relu_74, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, relu_76, convolution_80, squeeze_241, relu_77, convolution_81, squeeze_244, cat_10, convolution_82, squeeze_247, relu_79, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, relu_81, convolution_85, squeeze_256, relu_82, convolution_86, squeeze_259, relu_83, convolution_87, squeeze_262, relu_84, convolution_88, squeeze_265, cat_11, convolution_89, squeeze_268, relu_86, convolution_90, squeeze_271, relu_87, convolution_91, squeeze_274, relu_88, convolution_92, squeeze_277, relu_89, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, relu_91, convolution_95, squeeze_286, cat_12, convolution_96, squeeze_289, relu_93, getitem_210, getitem_211, convolution_97, squeeze_292, convolution_98, squeeze_295, relu_94, convolution_99, squeeze_298, relu_95, convolution_100, squeeze_301, relu_96, convolution_101, squeeze_304, relu_97, convolution_102, squeeze_307, relu_98, convolution_103, squeeze_310, cat_13, convolution_104, squeeze_313, clone, le, unsqueeze_422, le_1, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, le_8, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, le_15, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, le_22, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, le_29, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, le_36, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, le_43, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, unsqueeze_1022, le_50, unsqueeze_1034, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, unsqueeze_1094, unsqueeze_1106, le_57, unsqueeze_1118, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, le_64, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, le_71, unsqueeze_1298, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, le_78, unsqueeze_1382, unsqueeze_1394, unsqueeze_1406, unsqueeze_1418, unsqueeze_1430, unsqueeze_1442, unsqueeze_1454, le_85, unsqueeze_1466, unsqueeze_1478, unsqueeze_1490, unsqueeze_1502, unsqueeze_1514, unsqueeze_1526, unsqueeze_1538, unsqueeze_1550, le_92, unsqueeze_1562, unsqueeze_1574, unsqueeze_1586, unsqueeze_1598, unsqueeze_1610, unsqueeze_1622, unsqueeze_1634, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dla102', benchmark_compiled_module)
