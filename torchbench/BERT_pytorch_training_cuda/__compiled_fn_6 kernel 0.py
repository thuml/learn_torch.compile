
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


# kernel path: /tmp/torchinductor_youkaichao/ds/cdslub5636sph6cd345pinuezrba7afqv4kzv7gpb4bczroktf2e.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/fh/cfhgagitx25x432fft5hox7g7eydgqizvjvgt2ogbjrxaesn2bw4.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ii/ciiavibqi54exkkqh5f2s37qatkravtt5rln25oayoqcgzpsmskh.py
# Source Nodes: [l__mod___transformer_blocks_11_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
# l__mod___transformer_blocks_11_feed_forward_activation => add_84, erf_11, mul_58
triton_poi_fused_gelu_gelu_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/62/c62zemr6mntacjfjqimr3cohcw2lssqhw2cwjeg4eu4txiddoatk.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (393216*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caehorzmxdoi7a7djdwbbjyk2ch3klfqnklwl6bdcoak247leqgb.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzede7duyc4j5g5mt5kvedwnocm4ur5lpymvh5qcys4ictssolx.py
# Source Nodes: [add_71], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
# add_71 => add_82
triton_red_fused_add_div_mul_sum_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = 1e-06
        tmp6 = tmp4 + tmp5
        tmp7 = tmp0 / tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/24/c24e3mkohsrgftrhwevcp6rdkrtefli4cwqjd2rkmp5zyxe6yqwq.py
# Source Nodes: [add_71, mul_23, truediv_35], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
# add_71 => add_82
# mul_23 => mul_56
# truediv_35 => div_47
triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = -tmp0
    tmp4 = tmp2 * tmp3
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 / tmp7
    tmp9 = tmp8 / tmp7
    tmp10 = tmp1 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tmp0 / tmp7
    tmp16 = tmp15 * tmp2
    tmp17 = -tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = tmp22 + tmp16
    tmp24 = 0.0
    tmp25 = tmp5 == tmp24
    tmp26 = 2.0
    tmp27 = tmp5 * tmp26
    tmp28 = tmp14 / tmp27
    tmp29 = tl.where(tmp25, tmp24, tmp28)
    tmp30 = 0.002607561929595828
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31 * tmp3
    tmp33 = tmp23 + tmp32
    tmp34 = 768.0
    tmp35 = tmp21 / tmp34
    tmp36 = tmp33 + tmp35
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp36, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnx2i5irutoepzbsixvxdsakqmv7u3pufmyootokiufysworldjm.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192) % 12
    x3 = (xindex // 98304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (98304*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfecnkgi73h5yvnaq6xhvw7dv56glkglgndxgolrbyrpelnuw7tv.py
# Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
# eq => eq
triton_per_fused__softmax_backward_data_div_eq_masked_fill_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_div_eq_masked_fill_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 128
    x4 = (xindex // 1536)
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x2) + (16384*x4)), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tmp8 == tmp9
    tmp11 = tmp1 * tmp6
    tmp12 = tmp2 - tmp11
    tmp13 = 0.0
    tmp14 = tl.where(tmp10, tmp13, tmp12)
    tmp15 = 8.0
    tmp16 = tmp14 / tmp15
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp16, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cvefxy67llyqz75tiexbuwlkwk6shnfhjfm34hreedxiwmszb7aj.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 128)) + (8192*(x0 // 64)) + (98304*(x1 // 128)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvsgspzfh4ximhpyxeucoa2i2icz6yvnjg3mlbe6cbd6cpj4me7.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vs/cvswpcqwvoljqddfkjm6cbwvrgvassndpxwhrruiscj6l4f2ay5m.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((128*x1) + (98304*(y0 // 128)) + (y0 % 128)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (768*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdobpnos3ufztmusufyfogjmzhifjq6fauc7xewgz3pdtqu5f2fw.py
# Source Nodes: [add_68], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
# add_68 => add_79
triton_red_fused_add_div_mul_sum_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = 1e-06
        tmp10 = tmp8 + tmp9
        tmp11 = tmp4 / tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7s/c7sgang36jcqhanzwwpyrqg6a7s4oyhpi7g65yut5rrnen3lb2gu.py
# Source Nodes: [add_68, mul_22, truediv_33], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
# add_68 => add_79
# mul_22 => mul_55
# truediv_33 => div_44
triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = -tmp4
    tmp8 = tmp6 * tmp7
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 / tmp11
    tmp13 = tmp12 / tmp11
    tmp14 = tmp5 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp4 / tmp11
    tmp20 = tmp19 * tmp6
    tmp21 = -tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp27 = tmp26 + tmp20
    tmp28 = 0.0
    tmp29 = tmp9 == tmp28
    tmp30 = 2.0
    tmp31 = tmp9 * tmp30
    tmp32 = tmp18 / tmp31
    tmp33 = tl.where(tmp29, tmp28, tmp32)
    tmp34 = 0.002607561929595828
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35 * tmp7
    tmp37 = tmp27 + tmp36
    tmp38 = 768.0
    tmp39 = tmp25 / tmp38
    tmp40 = tmp37 + tmp39
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp40, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnp64axog2zl3jr2bz6q73j27u3ed5uspl3qbxg7qlp2u6jlapuu.py
# Source Nodes: [add_65, mul_21, truediv_32], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
# add_65 => add_75
# mul_21 => mul_51
# truediv_32 => div_43
triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = -tmp0
    tmp4 = tmp2 * tmp3
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 / tmp7
    tmp9 = tmp8 / tmp7
    tmp10 = tmp1 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tmp0 / tmp7
    tmp16 = tmp15 * tmp2
    tmp17 = -tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = tmp22 + tmp16
    tmp24 = 0.0
    tmp25 = tmp5 == tmp24
    tmp26 = 2.0
    tmp27 = tmp5 * tmp26
    tmp28 = tmp14 / tmp27
    tmp29 = tl.where(tmp25, tmp24, tmp28)
    tmp30 = 0.002607561929595828
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31 * tmp3
    tmp33 = tmp23 + tmp32
    tmp34 = 768.0
    tmp35 = tmp21 / tmp34
    tmp36 = tmp33 + tmp35
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp36, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ek/ceki4nvafj6rxvljpxbvg7mswcjtvmeciqbovimvfcp74rtxigka.py
# Source Nodes: [add_2, mul, truediv], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
# add_2 => add_2
# mul => mul
# truediv => div
triton_per_fused_add_div_embedding_dense_backward_eq_masked_fill_mul_neg_sum_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*i64', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_eq_masked_fill_mul_neg_sum_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp41 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = -tmp4
    tmp8 = tmp6 * tmp7
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 / tmp11
    tmp13 = tmp12 / tmp11
    tmp14 = tmp5 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp4 / tmp11
    tmp20 = tmp19 * tmp6
    tmp21 = -tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp27 = tmp26 + tmp20
    tmp28 = 0.0
    tmp29 = tmp9 == tmp28
    tmp30 = 2.0
    tmp31 = tmp9 * tmp30
    tmp32 = tmp18 / tmp31
    tmp33 = tl.where(tmp29, tmp28, tmp32)
    tmp34 = 0.002607561929595828
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35 * tmp7
    tmp37 = tmp27 + tmp36
    tmp38 = 768.0
    tmp39 = tmp25 / tmp38
    tmp40 = tmp37 + tmp39
    tmp42 = tl.full([1], 0, tl.int64)
    tmp43 = tmp41 == tmp42
    tmp44 = tl.where(tmp43, tmp28, tmp40)
    tmp46 = tmp45 == tmp42
    tmp47 = tl.where(tmp46, tmp28, tmp40)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp44, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp47, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7uluh6p3wzisegtjtltjsxnvcvgjlcakjpd35lzy2bxpzflbsec.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cufzvwkenezjakxjeeusqe4cf663tkwfl7llkhnkoq226fcaum4v.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15363840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_196, primals_197, unsqueeze_1, sqrt, sub, view, view_16, sqrt_1, sub_2, view_18, addmm_4, view_20, sqrt_2, sub_3, view_22, view_38, sqrt_3, sub_5, view_40, addmm_10, view_42, sqrt_4, sub_6, view_44, view_60, sqrt_5, sub_8, view_62, addmm_16, view_64, sqrt_6, sub_9, view_66, view_82, sqrt_7, sub_11, view_84, addmm_22, view_86, sqrt_8, sub_12, view_88, view_104, sqrt_9, sub_14, view_106, addmm_28, view_108, sqrt_10, sub_15, view_110, view_126, sqrt_11, sub_17, view_128, addmm_34, view_130, sqrt_12, sub_18, view_132, view_148, sqrt_13, sub_20, view_150, addmm_40, view_152, sqrt_14, sub_21, view_154, view_170, sqrt_15, sub_23, view_172, addmm_46, view_174, sqrt_16, sub_24, view_176, view_192, sqrt_17, sub_26, view_194, addmm_52, view_196, sqrt_18, sub_27, view_198, view_214, sqrt_19, sub_29, view_216, addmm_58, view_218, sqrt_20, sub_30, view_220, view_236, sqrt_21, sub_32, view_238, addmm_64, view_240, sqrt_22, sub_33, view_242, view_258, sqrt_23, sub_35, view_260, addmm_70, view_262, permute_132, permute_136, permute_140, permute_145, permute_146, alias_37, permute_147, permute_148, permute_151, permute_156, permute_161, permute_165, permute_169, permute_173, permute_178, permute_179, alias_40, permute_180, permute_181, permute_184, permute_189, permute_194, permute_198, permute_202, permute_206, permute_211, permute_212, alias_43, permute_213, permute_214, permute_217, permute_222, permute_227, permute_231, permute_235, permute_239, permute_244, permute_245, alias_46, permute_246, permute_247, permute_250, permute_255, permute_260, permute_264, permute_268, permute_272, permute_277, permute_278, alias_49, permute_279, permute_280, permute_283, permute_288, permute_293, permute_297, permute_301, permute_305, permute_310, permute_311, alias_52, permute_312, permute_313, permute_316, permute_321, permute_326, permute_330, permute_334, permute_338, permute_343, permute_344, alias_55, permute_345, permute_346, permute_349, permute_354, permute_359, permute_363, permute_367, permute_371, permute_376, permute_377, alias_58, permute_378, permute_379, permute_382, permute_387, permute_392, permute_396, permute_400, permute_404, permute_409, permute_410, alias_61, permute_411, permute_412, permute_415, permute_420, permute_425, permute_429, permute_433, permute_437, permute_442, permute_443, alias_64, permute_444, permute_445, permute_448, permute_453, permute_458, permute_462, permute_466, permute_470, permute_475, permute_476, alias_67, permute_477, permute_478, permute_481, permute_486, permute_491, permute_495, permute_499, permute_503, permute_508, permute_509, alias_70, permute_510, permute_511, permute_514, permute_519, permute_524, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_196, (4, 128), (128, 1))
    assert_size_stride(primals_197, (4, 128), (128, 1))
    assert_size_stride(unsqueeze_1, (4, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(sqrt, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(view_16, (512, 768), (768, 1))
    assert_size_stride(sqrt_1, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_2, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(addmm_4, (512, 3072), (3072, 1))
    assert_size_stride(view_20, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_2, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_3, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_22, (512, 768), (768, 1))
    assert_size_stride(view_38, (512, 768), (768, 1))
    assert_size_stride(sqrt_3, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_5, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_40, (512, 768), (768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(view_42, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_4, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_6, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_44, (512, 768), (768, 1))
    assert_size_stride(view_60, (512, 768), (768, 1))
    assert_size_stride(sqrt_5, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_8, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_62, (512, 768), (768, 1))
    assert_size_stride(addmm_16, (512, 3072), (3072, 1))
    assert_size_stride(view_64, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_6, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_9, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(view_82, (512, 768), (768, 1))
    assert_size_stride(sqrt_7, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_11, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_84, (512, 768), (768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(view_86, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_8, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_12, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_88, (512, 768), (768, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(sqrt_9, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_14, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_106, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 3072), (3072, 1))
    assert_size_stride(view_108, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_10, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_15, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_110, (512, 768), (768, 1))
    assert_size_stride(view_126, (512, 768), (768, 1))
    assert_size_stride(sqrt_11, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_17, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_128, (512, 768), (768, 1))
    assert_size_stride(addmm_34, (512, 3072), (3072, 1))
    assert_size_stride(view_130, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_12, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_18, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_132, (512, 768), (768, 1))
    assert_size_stride(view_148, (512, 768), (768, 1))
    assert_size_stride(sqrt_13, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_20, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_150, (512, 768), (768, 1))
    assert_size_stride(addmm_40, (512, 3072), (3072, 1))
    assert_size_stride(view_152, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_14, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_21, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_154, (512, 768), (768, 1))
    assert_size_stride(view_170, (512, 768), (768, 1))
    assert_size_stride(sqrt_15, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_23, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_172, (512, 768), (768, 1))
    assert_size_stride(addmm_46, (512, 3072), (3072, 1))
    assert_size_stride(view_174, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_16, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_24, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(view_192, (512, 768), (768, 1))
    assert_size_stride(sqrt_17, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_26, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_194, (512, 768), (768, 1))
    assert_size_stride(addmm_52, (512, 3072), (3072, 1))
    assert_size_stride(view_196, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_18, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_27, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_198, (512, 768), (768, 1))
    assert_size_stride(view_214, (512, 768), (768, 1))
    assert_size_stride(sqrt_19, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_29, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(addmm_58, (512, 3072), (3072, 1))
    assert_size_stride(view_218, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_20, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_30, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_220, (512, 768), (768, 1))
    assert_size_stride(view_236, (512, 768), (768, 1))
    assert_size_stride(sqrt_21, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_32, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_238, (512, 768), (768, 1))
    assert_size_stride(addmm_64, (512, 3072), (3072, 1))
    assert_size_stride(view_240, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_22, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_33, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_242, (512, 768), (768, 1))
    assert_size_stride(view_258, (512, 768), (768, 1))
    assert_size_stride(sqrt_23, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_35, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_260, (512, 768), (768, 1))
    assert_size_stride(addmm_70, (512, 3072), (3072, 1))
    assert_size_stride(view_262, (512, 3072), (3072, 1))
    assert_size_stride(permute_132, (768, 3072), (3072, 1))
    assert_size_stride(permute_136, (3072, 768), (768, 1))
    assert_size_stride(permute_140, (768, 768), (768, 1))
    assert_size_stride(permute_145, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_146, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_37, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_147, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_148, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_151, (768, 768), (768, 1))
    assert_size_stride(permute_156, (768, 768), (768, 1))
    assert_size_stride(permute_161, (768, 768), (768, 1))
    assert_size_stride(permute_165, (768, 3072), (3072, 1))
    assert_size_stride(permute_169, (3072, 768), (768, 1))
    assert_size_stride(permute_173, (768, 768), (768, 1))
    assert_size_stride(permute_178, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_179, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_40, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_180, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_181, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_184, (768, 768), (768, 1))
    assert_size_stride(permute_189, (768, 768), (768, 1))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(permute_198, (768, 3072), (3072, 1))
    assert_size_stride(permute_202, (3072, 768), (768, 1))
    assert_size_stride(permute_206, (768, 768), (768, 1))
    assert_size_stride(permute_211, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_212, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_43, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_213, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_214, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_217, (768, 768), (768, 1))
    assert_size_stride(permute_222, (768, 768), (768, 1))
    assert_size_stride(permute_227, (768, 768), (768, 1))
    assert_size_stride(permute_231, (768, 3072), (3072, 1))
    assert_size_stride(permute_235, (3072, 768), (768, 1))
    assert_size_stride(permute_239, (768, 768), (768, 1))
    assert_size_stride(permute_244, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_245, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_46, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_246, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_247, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_250, (768, 768), (768, 1))
    assert_size_stride(permute_255, (768, 768), (768, 1))
    assert_size_stride(permute_260, (768, 768), (768, 1))
    assert_size_stride(permute_264, (768, 3072), (3072, 1))
    assert_size_stride(permute_268, (3072, 768), (768, 1))
    assert_size_stride(permute_272, (768, 768), (768, 1))
    assert_size_stride(permute_277, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_278, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_49, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_279, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_280, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_283, (768, 768), (768, 1))
    assert_size_stride(permute_288, (768, 768), (768, 1))
    assert_size_stride(permute_293, (768, 768), (768, 1))
    assert_size_stride(permute_297, (768, 3072), (3072, 1))
    assert_size_stride(permute_301, (3072, 768), (768, 1))
    assert_size_stride(permute_305, (768, 768), (768, 1))
    assert_size_stride(permute_310, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_311, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_52, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_312, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_313, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_316, (768, 768), (768, 1))
    assert_size_stride(permute_321, (768, 768), (768, 1))
    assert_size_stride(permute_326, (768, 768), (768, 1))
    assert_size_stride(permute_330, (768, 3072), (3072, 1))
    assert_size_stride(permute_334, (3072, 768), (768, 1))
    assert_size_stride(permute_338, (768, 768), (768, 1))
    assert_size_stride(permute_343, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_344, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_55, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_345, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_346, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_349, (768, 768), (768, 1))
    assert_size_stride(permute_354, (768, 768), (768, 1))
    assert_size_stride(permute_359, (768, 768), (768, 1))
    assert_size_stride(permute_363, (768, 3072), (3072, 1))
    assert_size_stride(permute_367, (3072, 768), (768, 1))
    assert_size_stride(permute_371, (768, 768), (768, 1))
    assert_size_stride(permute_376, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_377, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_58, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_378, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_379, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_382, (768, 768), (768, 1))
    assert_size_stride(permute_387, (768, 768), (768, 1))
    assert_size_stride(permute_392, (768, 768), (768, 1))
    assert_size_stride(permute_396, (768, 3072), (3072, 1))
    assert_size_stride(permute_400, (3072, 768), (768, 1))
    assert_size_stride(permute_404, (768, 768), (768, 1))
    assert_size_stride(permute_409, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_410, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_61, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_411, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_412, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_415, (768, 768), (768, 1))
    assert_size_stride(permute_420, (768, 768), (768, 1))
    assert_size_stride(permute_425, (768, 768), (768, 1))
    assert_size_stride(permute_429, (768, 3072), (3072, 1))
    assert_size_stride(permute_433, (3072, 768), (768, 1))
    assert_size_stride(permute_437, (768, 768), (768, 1))
    assert_size_stride(permute_442, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_443, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_64, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_444, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_445, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_448, (768, 768), (768, 1))
    assert_size_stride(permute_453, (768, 768), (768, 1))
    assert_size_stride(permute_458, (768, 768), (768, 1))
    assert_size_stride(permute_462, (768, 3072), (3072, 1))
    assert_size_stride(permute_466, (3072, 768), (768, 1))
    assert_size_stride(permute_470, (768, 768), (768, 1))
    assert_size_stride(permute_475, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_476, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_67, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_477, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_478, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_481, (768, 768), (768, 1))
    assert_size_stride(permute_486, (768, 768), (768, 1))
    assert_size_stride(permute_491, (768, 768), (768, 1))
    assert_size_stride(permute_495, (768, 3072), (3072, 1))
    assert_size_stride(permute_499, (3072, 768), (768, 1))
    assert_size_stride(permute_503, (768, 768), (768, 1))
    assert_size_stride(permute_508, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_509, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_70, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_510, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_511, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_514, (768, 768), (768, 1))
    assert_size_stride(permute_519, (768, 768), (768, 1))
    assert_size_stride(permute_524, (768, 768), (768, 1))
    assert_size_stride(tangents_1, (4, 128, 768), (98304, 768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (512, 768), (768, 1), 0), permute_132, out=buf0)
        del permute_132
        buf1 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (768, 512), (1, 768), 0), view_262, out=buf1)
        del view_262
        buf2 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_red_fused_sum_0.run(tangents_1, buf2, 3072, 128, grid=grid(3072), stream=stream0)
        buf3 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf2, buf3, 768, 4, grid=grid(768), stream=stream0)
        buf4 = reinterpret_tensor(buf0, (4, 128, 3072), (393216, 3072, 1), 0); del buf0  # reuse
        # Source Nodes: [l__mod___transformer_blocks_11_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf4, addmm_70, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_70
        buf5 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 3072), (3072, 1), 0), permute_136, out=buf5)
        del permute_136
        buf6 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (3072, 512), (1, 3072), 0), view_260, out=buf6)
        del view_260
        buf7 = empty_strided((1, 3072, 4), (12288, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf4, buf7, 12288, 128, grid=grid(12288), stream=stream0)
        buf8 = reinterpret_tensor(buf2, (1, 3072), (3072, 1), 0); del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf7, buf8, 3072, 4, grid=grid(3072), stream=stream0)
        buf9 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_71], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf5, sqrt_23, sub_35, buf9, buf12, 3072, 128, grid=grid(3072), stream=stream0)
        buf10 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf9, buf10, 768, 4, grid=grid(768), stream=stream0)
        buf15 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_71, mul_23, truediv_35], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_6.run(buf5, primals_47, sub_35, sqrt_23, tangents_1, buf15, 512, 768, grid=grid(512), stream=stream0)
        del primals_47
        del sqrt_23
        del sub_35
        del tangents_1
        buf13 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_71], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf12, buf13, 768, 4, grid=grid(768), stream=stream0)
        buf16 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (512, 768), (768, 1), 0), permute_140, out=buf16)
        del permute_140
        buf17 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (768, 512), (1, 768), 0), view_258, out=buf17)
        del view_258
        buf18 = reinterpret_tensor(buf12, (1, 768, 4), (3072, 1, 768), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf15, buf18, 3072, 128, grid=grid(3072), stream=stream0)
        buf19 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf18, buf19, 768, 4, grid=grid(768), stream=stream0)
        buf20 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf16, buf20, 393216, grid=grid(393216), stream=stream0)
        buf21 = reinterpret_tensor(buf16, (48, 128, 64), (8192, 64, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_145, reinterpret_tensor(buf20, (48, 128, 64), (8192, 64, 1), 0), out=buf21)
        del permute_145
        buf22 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (48, 128, 64), (8192, 64, 1), 0), permute_146, out=buf22)
        del permute_146
        buf24 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf22, alias_37, unsqueeze_1, buf24, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_37
        buf25 = reinterpret_tensor(buf20, (48, 64, 128), (8192, 128, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_147, reinterpret_tensor(buf24, (48, 128, 128), (16384, 128, 1), 0), out=buf25)
        del permute_147
        buf26 = empty((48, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (48, 128, 128), (16384, 128, 1), 0), permute_148, out=buf26)
        del permute_148
        buf27 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf21, buf27, 393216, grid=grid(393216), stream=stream0)
        buf28 = reinterpret_tensor(buf21, (512, 768), (768, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf27, permute_151, out=buf28)
        del permute_151
        buf29 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (768, 512), (1, 768), 0), view_242, out=buf29)
        buf30 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf27, buf30, 3072, 128, grid=grid(3072), stream=stream0)
        buf31 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf30, buf31, 768, 4, grid=grid(768), stream=stream0)
        buf32 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf25, buf32, 512, 768, grid=grid(512, 768), stream=stream0)
        buf33 = reinterpret_tensor(buf25, (512, 768), (768, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf32, permute_156, out=buf33)
        del permute_156
        buf34 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (768, 512), (1, 768), 0), view_242, out=buf34)
        buf35 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf32, buf35, 3072, 128, grid=grid(3072), stream=stream0)
        buf36 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf35, buf36, 768, 4, grid=grid(768), stream=stream0)
        buf37 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf26, buf37, 393216, grid=grid(393216), stream=stream0)
        buf38 = reinterpret_tensor(buf26, (512, 768), (768, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf37, permute_161, out=buf38)
        del permute_161
        buf39 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (768, 512), (1, 768), 0), view_242, out=buf39)
        del view_242
        buf40 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf37, buf40, 3072, 128, grid=grid(3072), stream=stream0)
        buf41 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf40, buf41, 768, 4, grid=grid(768), stream=stream0)
        buf42 = reinterpret_tensor(buf40, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf40  # reuse
        buf46 = buf9; del buf9  # reuse
        # Source Nodes: [add_68], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf28, buf33, buf38, sqrt_22, sub_33, buf42, buf46, 3072, 128, grid=grid(3072), stream=stream0)
        buf43 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf42, buf43, 768, 4, grid=grid(768), stream=stream0)
        buf49 = buf15; del buf15  # reuse
        # Source Nodes: [add_68, mul_22, truediv_33], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf49, buf28, buf33, buf38, primals_45, sub_33, sqrt_22, 512, 768, grid=grid(512), stream=stream0)
        del primals_45
        del sqrt_22
        del sub_33
        buf47 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_68], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf46, buf47, 768, 4, grid=grid(768), stream=stream0)
        buf50 = reinterpret_tensor(buf4, (512, 3072), (3072, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (512, 768), (768, 1), 0), permute_165, out=buf50)
        del permute_165
        buf51 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (768, 512), (1, 768), 0), view_240, out=buf51)
        del view_240
        buf52 = reinterpret_tensor(buf46, (1, 768, 4), (3072, 1, 768), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf49, buf52, 3072, 128, grid=grid(3072), stream=stream0)
        buf53 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf52, buf53, 768, 4, grid=grid(768), stream=stream0)
        buf54 = reinterpret_tensor(buf50, (4, 128, 3072), (393216, 3072, 1), 0); del buf50  # reuse
        # Source Nodes: [l__mod___transformer_blocks_10_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf54, addmm_64, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_64
        buf55 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (512, 3072), (3072, 1), 0), permute_169, out=buf55)
        del permute_169
        buf56 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (3072, 512), (1, 3072), 0), view_238, out=buf56)
        del view_238
        buf57 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf54, buf57, 12288, 128, grid=grid(12288), stream=stream0)
        buf58 = reinterpret_tensor(buf52, (1, 3072), (3072, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf57, buf58, 3072, 4, grid=grid(3072), stream=stream0)
        buf59 = buf42; del buf42  # reuse
        buf62 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_65], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf55, sqrt_21, sub_32, buf59, buf62, 3072, 128, grid=grid(3072), stream=stream0)
        buf60 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf59, buf60, 768, 4, grid=grid(768), stream=stream0)
        buf65 = buf49; del buf49  # reuse
        # Source Nodes: [add_65, mul_21, truediv_32], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf65, buf55, primals_43, sub_32, sqrt_21, 512, 768, grid=grid(512), stream=stream0)
        del primals_43
        del sqrt_21
        del sub_32
        buf63 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_65], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf62, buf63, 768, 4, grid=grid(768), stream=stream0)
        buf66 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (512, 768), (768, 1), 0), permute_173, out=buf66)
        del permute_173
        buf67 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (768, 512), (1, 768), 0), view_236, out=buf67)
        del view_236
        buf68 = reinterpret_tensor(buf62, (1, 768, 4), (3072, 1, 768), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf65, buf68, 3072, 128, grid=grid(3072), stream=stream0)
        buf69 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf68, buf69, 768, 4, grid=grid(768), stream=stream0)
        buf70 = reinterpret_tensor(buf33, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf66, buf70, 393216, grid=grid(393216), stream=stream0)
        buf71 = reinterpret_tensor(buf66, (48, 128, 64), (8192, 64, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_178, reinterpret_tensor(buf70, (48, 128, 64), (8192, 64, 1), 0), out=buf71)
        del permute_178
        buf72 = reinterpret_tensor(buf24, (48, 128, 128), (16384, 128, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (48, 128, 64), (8192, 64, 1), 0), permute_179, out=buf72)
        del permute_179
        buf74 = reinterpret_tensor(buf22, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf22  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf72, alias_40, unsqueeze_1, buf74, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_40
        buf75 = reinterpret_tensor(buf70, (48, 64, 128), (8192, 128, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_180, reinterpret_tensor(buf74, (48, 128, 128), (16384, 128, 1), 0), out=buf75)
        del permute_180
        buf76 = reinterpret_tensor(buf28, (48, 128, 64), (8192, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf74, (48, 128, 128), (16384, 128, 1), 0), permute_181, out=buf76)
        del permute_181
        buf77 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf71, buf77, 393216, grid=grid(393216), stream=stream0)
        buf78 = reinterpret_tensor(buf71, (512, 768), (768, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf77, permute_184, out=buf78)
        del permute_184
        buf79 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (768, 512), (1, 768), 0), view_220, out=buf79)
        buf80 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf77, buf80, 3072, 128, grid=grid(3072), stream=stream0)
        buf81 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf80, buf81, 768, 4, grid=grid(768), stream=stream0)
        buf82 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf75, buf82, 512, 768, grid=grid(512, 768), stream=stream0)
        buf83 = reinterpret_tensor(buf75, (512, 768), (768, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf82, permute_189, out=buf83)
        del permute_189
        buf84 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (768, 512), (1, 768), 0), view_220, out=buf84)
        buf85 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf82, buf85, 3072, 128, grid=grid(3072), stream=stream0)
        buf86 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf85, buf86, 768, 4, grid=grid(768), stream=stream0)
        buf87 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf76, buf87, 393216, grid=grid(393216), stream=stream0)
        buf88 = reinterpret_tensor(buf76, (512, 768), (768, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf87, permute_194, out=buf88)
        del permute_194
        buf89 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (768, 512), (1, 768), 0), view_220, out=buf89)
        del view_220
        buf90 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf87, buf90, 3072, 128, grid=grid(3072), stream=stream0)
        buf91 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf90, buf91, 768, 4, grid=grid(768), stream=stream0)
        buf92 = reinterpret_tensor(buf90, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf90  # reuse
        buf96 = buf59; del buf59  # reuse
        # Source Nodes: [add_62], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf78, buf83, buf88, sqrt_20, sub_30, buf92, buf96, 3072, 128, grid=grid(3072), stream=stream0)
        buf93 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf92, buf93, 768, 4, grid=grid(768), stream=stream0)
        buf99 = buf65; del buf65  # reuse
        # Source Nodes: [add_62, mul_20, truediv_30], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf99, buf78, buf83, buf88, primals_41, sub_30, sqrt_20, 512, 768, grid=grid(512), stream=stream0)
        del primals_41
        del sqrt_20
        del sub_30
        buf97 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_62], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf96, buf97, 768, 4, grid=grid(768), stream=stream0)
        buf100 = reinterpret_tensor(buf54, (512, 3072), (3072, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 768), (768, 1), 0), permute_198, out=buf100)
        del permute_198
        buf101 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (768, 512), (1, 768), 0), view_218, out=buf101)
        del view_218
        buf102 = reinterpret_tensor(buf96, (1, 768, 4), (3072, 1, 768), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf99, buf102, 3072, 128, grid=grid(3072), stream=stream0)
        buf103 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf102, buf103, 768, 4, grid=grid(768), stream=stream0)
        buf104 = reinterpret_tensor(buf100, (4, 128, 3072), (393216, 3072, 1), 0); del buf100  # reuse
        # Source Nodes: [l__mod___transformer_blocks_9_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf104, addmm_58, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_58
        buf105 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (512, 3072), (3072, 1), 0), permute_202, out=buf105)
        del permute_202
        buf106 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (3072, 512), (1, 3072), 0), view_216, out=buf106)
        del view_216
        buf107 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf104, buf107, 12288, 128, grid=grid(12288), stream=stream0)
        buf108 = reinterpret_tensor(buf102, (1, 3072), (3072, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf107, buf108, 3072, 4, grid=grid(3072), stream=stream0)
        buf109 = buf92; del buf92  # reuse
        buf112 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_59], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf105, sqrt_19, sub_29, buf109, buf112, 3072, 128, grid=grid(3072), stream=stream0)
        buf110 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf109, buf110, 768, 4, grid=grid(768), stream=stream0)
        buf115 = buf99; del buf99  # reuse
        # Source Nodes: [add_59, mul_19, truediv_29], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf115, buf105, primals_39, sub_29, sqrt_19, 512, 768, grid=grid(512), stream=stream0)
        del primals_39
        del sqrt_19
        del sub_29
        buf113 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_59], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf112, buf113, 768, 4, grid=grid(768), stream=stream0)
        buf116 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (512, 768), (768, 1), 0), permute_206, out=buf116)
        del permute_206
        buf117 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (768, 512), (1, 768), 0), view_214, out=buf117)
        del view_214
        buf118 = reinterpret_tensor(buf112, (1, 768, 4), (3072, 1, 768), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf115, buf118, 3072, 128, grid=grid(3072), stream=stream0)
        buf119 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf118, buf119, 768, 4, grid=grid(768), stream=stream0)
        buf120 = reinterpret_tensor(buf83, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf116, buf120, 393216, grid=grid(393216), stream=stream0)
        buf121 = reinterpret_tensor(buf116, (48, 128, 64), (8192, 64, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_211, reinterpret_tensor(buf120, (48, 128, 64), (8192, 64, 1), 0), out=buf121)
        del permute_211
        buf122 = reinterpret_tensor(buf74, (48, 128, 128), (16384, 128, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (48, 128, 64), (8192, 64, 1), 0), permute_212, out=buf122)
        del permute_212
        buf124 = reinterpret_tensor(buf72, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf72  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf122, alias_43, unsqueeze_1, buf124, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_43
        buf125 = reinterpret_tensor(buf120, (48, 64, 128), (8192, 128, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_213, reinterpret_tensor(buf124, (48, 128, 128), (16384, 128, 1), 0), out=buf125)
        del permute_213
        buf126 = reinterpret_tensor(buf78, (48, 128, 64), (8192, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf124, (48, 128, 128), (16384, 128, 1), 0), permute_214, out=buf126)
        del permute_214
        buf127 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf121, buf127, 393216, grid=grid(393216), stream=stream0)
        buf128 = reinterpret_tensor(buf121, (512, 768), (768, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf127, permute_217, out=buf128)
        del permute_217
        buf129 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (768, 512), (1, 768), 0), view_198, out=buf129)
        buf130 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf127, buf130, 3072, 128, grid=grid(3072), stream=stream0)
        buf131 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf130, buf131, 768, 4, grid=grid(768), stream=stream0)
        buf132 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf125, buf132, 512, 768, grid=grid(512, 768), stream=stream0)
        buf133 = reinterpret_tensor(buf125, (512, 768), (768, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf132, permute_222, out=buf133)
        del permute_222
        buf134 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (768, 512), (1, 768), 0), view_198, out=buf134)
        buf135 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf132, buf135, 3072, 128, grid=grid(3072), stream=stream0)
        buf136 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf135, buf136, 768, 4, grid=grid(768), stream=stream0)
        buf137 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf126, buf137, 393216, grid=grid(393216), stream=stream0)
        buf138 = reinterpret_tensor(buf126, (512, 768), (768, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf137, permute_227, out=buf138)
        del permute_227
        buf139 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (768, 512), (1, 768), 0), view_198, out=buf139)
        del view_198
        buf140 = buf135; del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf137, buf140, 3072, 128, grid=grid(3072), stream=stream0)
        buf141 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf140, buf141, 768, 4, grid=grid(768), stream=stream0)
        buf142 = reinterpret_tensor(buf140, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf140  # reuse
        buf146 = buf109; del buf109  # reuse
        # Source Nodes: [add_56], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf128, buf133, buf138, sqrt_18, sub_27, buf142, buf146, 3072, 128, grid=grid(3072), stream=stream0)
        buf143 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf142, buf143, 768, 4, grid=grid(768), stream=stream0)
        buf149 = buf115; del buf115  # reuse
        # Source Nodes: [add_56, mul_18, truediv_27], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf149, buf128, buf133, buf138, primals_37, sub_27, sqrt_18, 512, 768, grid=grid(512), stream=stream0)
        del primals_37
        del sqrt_18
        del sub_27
        buf147 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_56], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf146, buf147, 768, 4, grid=grid(768), stream=stream0)
        buf150 = reinterpret_tensor(buf104, (512, 3072), (3072, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (512, 768), (768, 1), 0), permute_231, out=buf150)
        del permute_231
        buf151 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (768, 512), (1, 768), 0), view_196, out=buf151)
        del view_196
        buf152 = reinterpret_tensor(buf146, (1, 768, 4), (3072, 1, 768), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf149, buf152, 3072, 128, grid=grid(3072), stream=stream0)
        buf153 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf152, buf153, 768, 4, grid=grid(768), stream=stream0)
        buf154 = reinterpret_tensor(buf150, (4, 128, 3072), (393216, 3072, 1), 0); del buf150  # reuse
        # Source Nodes: [l__mod___transformer_blocks_8_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf154, addmm_52, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_52
        buf155 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (512, 3072), (3072, 1), 0), permute_235, out=buf155)
        del permute_235
        buf156 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (3072, 512), (1, 3072), 0), view_194, out=buf156)
        del view_194
        buf157 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf154, buf157, 12288, 128, grid=grid(12288), stream=stream0)
        buf158 = reinterpret_tensor(buf152, (1, 3072), (3072, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf157, buf158, 3072, 4, grid=grid(3072), stream=stream0)
        buf159 = buf142; del buf142  # reuse
        buf162 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_53], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf155, sqrt_17, sub_26, buf159, buf162, 3072, 128, grid=grid(3072), stream=stream0)
        buf160 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf159, buf160, 768, 4, grid=grid(768), stream=stream0)
        buf165 = buf149; del buf149  # reuse
        # Source Nodes: [add_53, mul_17, truediv_26], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf165, buf155, primals_35, sub_26, sqrt_17, 512, 768, grid=grid(512), stream=stream0)
        del primals_35
        del sqrt_17
        del sub_26
        buf163 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_53], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf162, buf163, 768, 4, grid=grid(768), stream=stream0)
        buf166 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (512, 768), (768, 1), 0), permute_239, out=buf166)
        del permute_239
        buf167 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (768, 512), (1, 768), 0), view_192, out=buf167)
        del view_192
        buf168 = reinterpret_tensor(buf162, (1, 768, 4), (3072, 1, 768), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf165, buf168, 3072, 128, grid=grid(3072), stream=stream0)
        buf169 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf168, buf169, 768, 4, grid=grid(768), stream=stream0)
        buf170 = reinterpret_tensor(buf133, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf166, buf170, 393216, grid=grid(393216), stream=stream0)
        buf171 = reinterpret_tensor(buf166, (48, 128, 64), (8192, 64, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_244, reinterpret_tensor(buf170, (48, 128, 64), (8192, 64, 1), 0), out=buf171)
        del permute_244
        buf172 = reinterpret_tensor(buf124, (48, 128, 128), (16384, 128, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf170, (48, 128, 64), (8192, 64, 1), 0), permute_245, out=buf172)
        del permute_245
        buf174 = reinterpret_tensor(buf122, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf122  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf172, alias_46, unsqueeze_1, buf174, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_46
        buf175 = reinterpret_tensor(buf170, (48, 64, 128), (8192, 128, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_246, reinterpret_tensor(buf174, (48, 128, 128), (16384, 128, 1), 0), out=buf175)
        del permute_246
        buf176 = reinterpret_tensor(buf128, (48, 128, 64), (8192, 64, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf174, (48, 128, 128), (16384, 128, 1), 0), permute_247, out=buf176)
        del permute_247
        buf177 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf171, buf177, 393216, grid=grid(393216), stream=stream0)
        buf178 = reinterpret_tensor(buf171, (512, 768), (768, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf177, permute_250, out=buf178)
        del permute_250
        buf179 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (768, 512), (1, 768), 0), view_176, out=buf179)
        buf180 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf177, buf180, 3072, 128, grid=grid(3072), stream=stream0)
        buf181 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf180, buf181, 768, 4, grid=grid(768), stream=stream0)
        buf182 = buf177; del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf175, buf182, 512, 768, grid=grid(512, 768), stream=stream0)
        buf183 = reinterpret_tensor(buf175, (512, 768), (768, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf182, permute_255, out=buf183)
        del permute_255
        buf184 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (768, 512), (1, 768), 0), view_176, out=buf184)
        buf185 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf182, buf185, 3072, 128, grid=grid(3072), stream=stream0)
        buf186 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf185, buf186, 768, 4, grid=grid(768), stream=stream0)
        buf187 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf176, buf187, 393216, grid=grid(393216), stream=stream0)
        buf188 = reinterpret_tensor(buf176, (512, 768), (768, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf187, permute_260, out=buf188)
        del permute_260
        buf189 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (768, 512), (1, 768), 0), view_176, out=buf189)
        del view_176
        buf190 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf187, buf190, 3072, 128, grid=grid(3072), stream=stream0)
        buf191 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf190, buf191, 768, 4, grid=grid(768), stream=stream0)
        buf192 = reinterpret_tensor(buf190, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf190  # reuse
        buf196 = buf159; del buf159  # reuse
        # Source Nodes: [add_50], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf178, buf183, buf188, sqrt_16, sub_24, buf192, buf196, 3072, 128, grid=grid(3072), stream=stream0)
        buf193 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf192, buf193, 768, 4, grid=grid(768), stream=stream0)
        buf199 = buf165; del buf165  # reuse
        # Source Nodes: [add_50, mul_16, truediv_24], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf199, buf178, buf183, buf188, primals_33, sub_24, sqrt_16, 512, 768, grid=grid(512), stream=stream0)
        del primals_33
        del sqrt_16
        del sub_24
        buf197 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_50], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf196, buf197, 768, 4, grid=grid(768), stream=stream0)
        buf200 = reinterpret_tensor(buf154, (512, 3072), (3072, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (512, 768), (768, 1), 0), permute_264, out=buf200)
        del permute_264
        buf201 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (768, 512), (1, 768), 0), view_174, out=buf201)
        del view_174
        buf202 = reinterpret_tensor(buf196, (1, 768, 4), (3072, 1, 768), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf199, buf202, 3072, 128, grid=grid(3072), stream=stream0)
        buf203 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf202, buf203, 768, 4, grid=grid(768), stream=stream0)
        buf204 = reinterpret_tensor(buf200, (4, 128, 3072), (393216, 3072, 1), 0); del buf200  # reuse
        # Source Nodes: [l__mod___transformer_blocks_7_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf204, addmm_46, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_46
        buf205 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (512, 3072), (3072, 1), 0), permute_268, out=buf205)
        del permute_268
        buf206 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (3072, 512), (1, 3072), 0), view_172, out=buf206)
        del view_172
        buf207 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf204, buf207, 12288, 128, grid=grid(12288), stream=stream0)
        buf208 = reinterpret_tensor(buf202, (1, 3072), (3072, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf207, buf208, 3072, 4, grid=grid(3072), stream=stream0)
        buf209 = buf192; del buf192  # reuse
        buf212 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_47], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf205, sqrt_15, sub_23, buf209, buf212, 3072, 128, grid=grid(3072), stream=stream0)
        buf210 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf209, buf210, 768, 4, grid=grid(768), stream=stream0)
        buf215 = buf199; del buf199  # reuse
        # Source Nodes: [add_47, mul_15, truediv_23], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf215, buf205, primals_31, sub_23, sqrt_15, 512, 768, grid=grid(512), stream=stream0)
        del primals_31
        del sqrt_15
        del sub_23
        buf213 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_47], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf212, buf213, 768, 4, grid=grid(768), stream=stream0)
        buf216 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (512, 768), (768, 1), 0), permute_272, out=buf216)
        del permute_272
        buf217 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (768, 512), (1, 768), 0), view_170, out=buf217)
        del view_170
        buf218 = reinterpret_tensor(buf212, (1, 768, 4), (3072, 1, 768), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf215, buf218, 3072, 128, grid=grid(3072), stream=stream0)
        buf219 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf218, buf219, 768, 4, grid=grid(768), stream=stream0)
        buf220 = reinterpret_tensor(buf183, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf216, buf220, 393216, grid=grid(393216), stream=stream0)
        buf221 = reinterpret_tensor(buf216, (48, 128, 64), (8192, 64, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_277, reinterpret_tensor(buf220, (48, 128, 64), (8192, 64, 1), 0), out=buf221)
        del permute_277
        buf222 = reinterpret_tensor(buf174, (48, 128, 128), (16384, 128, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (48, 128, 64), (8192, 64, 1), 0), permute_278, out=buf222)
        del permute_278
        buf224 = reinterpret_tensor(buf172, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf172  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf222, alias_49, unsqueeze_1, buf224, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_49
        buf225 = reinterpret_tensor(buf220, (48, 64, 128), (8192, 128, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_279, reinterpret_tensor(buf224, (48, 128, 128), (16384, 128, 1), 0), out=buf225)
        del permute_279
        buf226 = reinterpret_tensor(buf178, (48, 128, 64), (8192, 64, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (48, 128, 128), (16384, 128, 1), 0), permute_280, out=buf226)
        del permute_280
        buf227 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf221, buf227, 393216, grid=grid(393216), stream=stream0)
        buf228 = reinterpret_tensor(buf221, (512, 768), (768, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf227, permute_283, out=buf228)
        del permute_283
        buf229 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (768, 512), (1, 768), 0), view_154, out=buf229)
        buf230 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf227, buf230, 3072, 128, grid=grid(3072), stream=stream0)
        buf231 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf230, buf231, 768, 4, grid=grid(768), stream=stream0)
        buf232 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf225, buf232, 512, 768, grid=grid(512, 768), stream=stream0)
        buf233 = reinterpret_tensor(buf225, (512, 768), (768, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf232, permute_288, out=buf233)
        del permute_288
        buf234 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (768, 512), (1, 768), 0), view_154, out=buf234)
        buf235 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf232, buf235, 3072, 128, grid=grid(3072), stream=stream0)
        buf236 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf235, buf236, 768, 4, grid=grid(768), stream=stream0)
        buf237 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf226, buf237, 393216, grid=grid(393216), stream=stream0)
        buf238 = reinterpret_tensor(buf226, (512, 768), (768, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf237, permute_293, out=buf238)
        del permute_293
        buf239 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (768, 512), (1, 768), 0), view_154, out=buf239)
        del view_154
        buf240 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf237, buf240, 3072, 128, grid=grid(3072), stream=stream0)
        buf241 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf240, buf241, 768, 4, grid=grid(768), stream=stream0)
        buf242 = reinterpret_tensor(buf240, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf240  # reuse
        buf246 = buf209; del buf209  # reuse
        # Source Nodes: [add_44], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf228, buf233, buf238, sqrt_14, sub_21, buf242, buf246, 3072, 128, grid=grid(3072), stream=stream0)
        buf243 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf242, buf243, 768, 4, grid=grid(768), stream=stream0)
        buf249 = buf215; del buf215  # reuse
        # Source Nodes: [add_44, mul_14, truediv_21], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf249, buf228, buf233, buf238, primals_29, sub_21, sqrt_14, 512, 768, grid=grid(512), stream=stream0)
        del primals_29
        del sqrt_14
        del sub_21
        buf247 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_44], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf246, buf247, 768, 4, grid=grid(768), stream=stream0)
        buf250 = reinterpret_tensor(buf204, (512, 3072), (3072, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 768), (768, 1), 0), permute_297, out=buf250)
        del permute_297
        buf251 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (768, 512), (1, 768), 0), view_152, out=buf251)
        del view_152
        buf252 = reinterpret_tensor(buf246, (1, 768, 4), (3072, 1, 768), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf249, buf252, 3072, 128, grid=grid(3072), stream=stream0)
        buf253 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf252, buf253, 768, 4, grid=grid(768), stream=stream0)
        buf254 = reinterpret_tensor(buf250, (4, 128, 3072), (393216, 3072, 1), 0); del buf250  # reuse
        # Source Nodes: [l__mod___transformer_blocks_6_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf254, addmm_40, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_40
        buf255 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (512, 3072), (3072, 1), 0), permute_301, out=buf255)
        del permute_301
        buf256 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (3072, 512), (1, 3072), 0), view_150, out=buf256)
        del view_150
        buf257 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf254, buf257, 12288, 128, grid=grid(12288), stream=stream0)
        buf258 = reinterpret_tensor(buf252, (1, 3072), (3072, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf257, buf258, 3072, 4, grid=grid(3072), stream=stream0)
        buf259 = buf242; del buf242  # reuse
        buf262 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_41], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf255, sqrt_13, sub_20, buf259, buf262, 3072, 128, grid=grid(3072), stream=stream0)
        buf260 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf259, buf260, 768, 4, grid=grid(768), stream=stream0)
        buf265 = buf249; del buf249  # reuse
        # Source Nodes: [add_41, mul_13, truediv_20], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf265, buf255, primals_27, sub_20, sqrt_13, 512, 768, grid=grid(512), stream=stream0)
        del primals_27
        del sqrt_13
        del sub_20
        buf263 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_41], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf262, buf263, 768, 4, grid=grid(768), stream=stream0)
        buf266 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (512, 768), (768, 1), 0), permute_305, out=buf266)
        del permute_305
        buf267 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (768, 512), (1, 768), 0), view_148, out=buf267)
        del view_148
        buf268 = reinterpret_tensor(buf262, (1, 768, 4), (3072, 1, 768), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf265, buf268, 3072, 128, grid=grid(3072), stream=stream0)
        buf269 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf268, buf269, 768, 4, grid=grid(768), stream=stream0)
        buf270 = reinterpret_tensor(buf233, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf266, buf270, 393216, grid=grid(393216), stream=stream0)
        buf271 = reinterpret_tensor(buf266, (48, 128, 64), (8192, 64, 1), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_310, reinterpret_tensor(buf270, (48, 128, 64), (8192, 64, 1), 0), out=buf271)
        del permute_310
        buf272 = reinterpret_tensor(buf224, (48, 128, 128), (16384, 128, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (48, 128, 64), (8192, 64, 1), 0), permute_311, out=buf272)
        del permute_311
        buf274 = reinterpret_tensor(buf222, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf222  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf272, alias_52, unsqueeze_1, buf274, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_52
        buf275 = reinterpret_tensor(buf270, (48, 64, 128), (8192, 128, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_312, reinterpret_tensor(buf274, (48, 128, 128), (16384, 128, 1), 0), out=buf275)
        del permute_312
        buf276 = reinterpret_tensor(buf228, (48, 128, 64), (8192, 64, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf274, (48, 128, 128), (16384, 128, 1), 0), permute_313, out=buf276)
        del permute_313
        buf277 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf271, buf277, 393216, grid=grid(393216), stream=stream0)
        buf278 = reinterpret_tensor(buf271, (512, 768), (768, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf277, permute_316, out=buf278)
        del permute_316
        buf279 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (768, 512), (1, 768), 0), view_132, out=buf279)
        buf280 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf277, buf280, 3072, 128, grid=grid(3072), stream=stream0)
        buf281 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf280, buf281, 768, 4, grid=grid(768), stream=stream0)
        buf282 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf275, buf282, 512, 768, grid=grid(512, 768), stream=stream0)
        buf283 = reinterpret_tensor(buf275, (512, 768), (768, 1), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf282, permute_321, out=buf283)
        del permute_321
        buf284 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (768, 512), (1, 768), 0), view_132, out=buf284)
        buf285 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf282, buf285, 3072, 128, grid=grid(3072), stream=stream0)
        buf286 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf285, buf286, 768, 4, grid=grid(768), stream=stream0)
        buf287 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf276, buf287, 393216, grid=grid(393216), stream=stream0)
        buf288 = reinterpret_tensor(buf276, (512, 768), (768, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf287, permute_326, out=buf288)
        del permute_326
        buf289 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (768, 512), (1, 768), 0), view_132, out=buf289)
        del view_132
        buf290 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf287, buf290, 3072, 128, grid=grid(3072), stream=stream0)
        buf291 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf290, buf291, 768, 4, grid=grid(768), stream=stream0)
        buf292 = reinterpret_tensor(buf290, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf290  # reuse
        buf296 = buf259; del buf259  # reuse
        # Source Nodes: [add_38], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf278, buf283, buf288, sqrt_12, sub_18, buf292, buf296, 3072, 128, grid=grid(3072), stream=stream0)
        buf293 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf292, buf293, 768, 4, grid=grid(768), stream=stream0)
        buf299 = buf265; del buf265  # reuse
        # Source Nodes: [add_38, mul_12, truediv_18], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf299, buf278, buf283, buf288, primals_25, sub_18, sqrt_12, 512, 768, grid=grid(512), stream=stream0)
        del primals_25
        del sqrt_12
        del sub_18
        buf297 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf296, buf297, 768, 4, grid=grid(768), stream=stream0)
        buf300 = reinterpret_tensor(buf254, (512, 3072), (3072, 1), 0); del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (512, 768), (768, 1), 0), permute_330, out=buf300)
        del permute_330
        buf301 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (768, 512), (1, 768), 0), view_130, out=buf301)
        del view_130
        buf302 = reinterpret_tensor(buf296, (1, 768, 4), (3072, 1, 768), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf299, buf302, 3072, 128, grid=grid(3072), stream=stream0)
        buf303 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf302, buf303, 768, 4, grid=grid(768), stream=stream0)
        buf304 = reinterpret_tensor(buf300, (4, 128, 3072), (393216, 3072, 1), 0); del buf300  # reuse
        # Source Nodes: [l__mod___transformer_blocks_5_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf304, addmm_34, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_34
        buf305 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (512, 3072), (3072, 1), 0), permute_334, out=buf305)
        del permute_334
        buf306 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (3072, 512), (1, 3072), 0), view_128, out=buf306)
        del view_128
        buf307 = buf257; del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf304, buf307, 12288, 128, grid=grid(12288), stream=stream0)
        buf308 = reinterpret_tensor(buf302, (1, 3072), (3072, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf307, buf308, 3072, 4, grid=grid(3072), stream=stream0)
        buf309 = buf292; del buf292  # reuse
        buf312 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_35], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf305, sqrt_11, sub_17, buf309, buf312, 3072, 128, grid=grid(3072), stream=stream0)
        buf310 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf309, buf310, 768, 4, grid=grid(768), stream=stream0)
        buf315 = buf299; del buf299  # reuse
        # Source Nodes: [add_35, mul_11, truediv_17], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf315, buf305, primals_23, sub_17, sqrt_11, 512, 768, grid=grid(512), stream=stream0)
        del primals_23
        del sqrt_11
        del sub_17
        buf313 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_35], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf312, buf313, 768, 4, grid=grid(768), stream=stream0)
        buf316 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (512, 768), (768, 1), 0), permute_338, out=buf316)
        del permute_338
        buf317 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (768, 512), (1, 768), 0), view_126, out=buf317)
        del view_126
        buf318 = reinterpret_tensor(buf312, (1, 768, 4), (3072, 1, 768), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf315, buf318, 3072, 128, grid=grid(3072), stream=stream0)
        buf319 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf318, buf319, 768, 4, grid=grid(768), stream=stream0)
        buf320 = reinterpret_tensor(buf283, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf316, buf320, 393216, grid=grid(393216), stream=stream0)
        buf321 = reinterpret_tensor(buf316, (48, 128, 64), (8192, 64, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_343, reinterpret_tensor(buf320, (48, 128, 64), (8192, 64, 1), 0), out=buf321)
        del permute_343
        buf322 = reinterpret_tensor(buf274, (48, 128, 128), (16384, 128, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf320, (48, 128, 64), (8192, 64, 1), 0), permute_344, out=buf322)
        del permute_344
        buf324 = reinterpret_tensor(buf272, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf272  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf322, alias_55, unsqueeze_1, buf324, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_55
        buf325 = reinterpret_tensor(buf320, (48, 64, 128), (8192, 128, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_345, reinterpret_tensor(buf324, (48, 128, 128), (16384, 128, 1), 0), out=buf325)
        del permute_345
        buf326 = reinterpret_tensor(buf278, (48, 128, 64), (8192, 64, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf324, (48, 128, 128), (16384, 128, 1), 0), permute_346, out=buf326)
        del permute_346
        buf327 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf321, buf327, 393216, grid=grid(393216), stream=stream0)
        buf328 = reinterpret_tensor(buf321, (512, 768), (768, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf327, permute_349, out=buf328)
        del permute_349
        buf329 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (768, 512), (1, 768), 0), view_110, out=buf329)
        buf330 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf327, buf330, 3072, 128, grid=grid(3072), stream=stream0)
        buf331 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf330, buf331, 768, 4, grid=grid(768), stream=stream0)
        buf332 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf325, buf332, 512, 768, grid=grid(512, 768), stream=stream0)
        buf333 = reinterpret_tensor(buf325, (512, 768), (768, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf332, permute_354, out=buf333)
        del permute_354
        buf334 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (768, 512), (1, 768), 0), view_110, out=buf334)
        buf335 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf332, buf335, 3072, 128, grid=grid(3072), stream=stream0)
        buf336 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf335, buf336, 768, 4, grid=grid(768), stream=stream0)
        buf337 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf326, buf337, 393216, grid=grid(393216), stream=stream0)
        buf338 = reinterpret_tensor(buf326, (512, 768), (768, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf337, permute_359, out=buf338)
        del permute_359
        buf339 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (768, 512), (1, 768), 0), view_110, out=buf339)
        del view_110
        buf340 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf337, buf340, 3072, 128, grid=grid(3072), stream=stream0)
        buf341 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf340, buf341, 768, 4, grid=grid(768), stream=stream0)
        buf342 = reinterpret_tensor(buf340, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf340  # reuse
        buf346 = buf309; del buf309  # reuse
        # Source Nodes: [add_32], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf328, buf333, buf338, sqrt_10, sub_15, buf342, buf346, 3072, 128, grid=grid(3072), stream=stream0)
        buf343 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf342, buf343, 768, 4, grid=grid(768), stream=stream0)
        buf349 = buf315; del buf315  # reuse
        # Source Nodes: [add_32, mul_10, truediv_15], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf349, buf328, buf333, buf338, primals_21, sub_15, sqrt_10, 512, 768, grid=grid(512), stream=stream0)
        del primals_21
        del sqrt_10
        del sub_15
        buf347 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_32], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf346, buf347, 768, 4, grid=grid(768), stream=stream0)
        buf350 = reinterpret_tensor(buf304, (512, 3072), (3072, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (512, 768), (768, 1), 0), permute_363, out=buf350)
        del permute_363
        buf351 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (768, 512), (1, 768), 0), view_108, out=buf351)
        del view_108
        buf352 = reinterpret_tensor(buf346, (1, 768, 4), (3072, 1, 768), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf349, buf352, 3072, 128, grid=grid(3072), stream=stream0)
        buf353 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf352, buf353, 768, 4, grid=grid(768), stream=stream0)
        buf354 = reinterpret_tensor(buf350, (4, 128, 3072), (393216, 3072, 1), 0); del buf350  # reuse
        # Source Nodes: [l__mod___transformer_blocks_4_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf354, addmm_28, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_28
        buf355 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf354, (512, 3072), (3072, 1), 0), permute_367, out=buf355)
        del permute_367
        buf356 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf354, (3072, 512), (1, 3072), 0), view_106, out=buf356)
        del view_106
        buf357 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf354, buf357, 12288, 128, grid=grid(12288), stream=stream0)
        buf358 = reinterpret_tensor(buf352, (1, 3072), (3072, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf357, buf358, 3072, 4, grid=grid(3072), stream=stream0)
        buf359 = buf342; del buf342  # reuse
        buf362 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf355, sqrt_9, sub_14, buf359, buf362, 3072, 128, grid=grid(3072), stream=stream0)
        buf360 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf359, buf360, 768, 4, grid=grid(768), stream=stream0)
        buf365 = buf349; del buf349  # reuse
        # Source Nodes: [add_29, mul_9, truediv_14], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf365, buf355, primals_19, sub_14, sqrt_9, 512, 768, grid=grid(512), stream=stream0)
        del primals_19
        del sqrt_9
        del sub_14
        buf363 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf362, buf363, 768, 4, grid=grid(768), stream=stream0)
        buf366 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (512, 768), (768, 1), 0), permute_371, out=buf366)
        del permute_371
        buf367 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (768, 512), (1, 768), 0), view_104, out=buf367)
        del view_104
        buf368 = reinterpret_tensor(buf362, (1, 768, 4), (3072, 1, 768), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf365, buf368, 3072, 128, grid=grid(3072), stream=stream0)
        buf369 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf368, buf369, 768, 4, grid=grid(768), stream=stream0)
        buf370 = reinterpret_tensor(buf333, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf366, buf370, 393216, grid=grid(393216), stream=stream0)
        buf371 = reinterpret_tensor(buf366, (48, 128, 64), (8192, 64, 1), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_376, reinterpret_tensor(buf370, (48, 128, 64), (8192, 64, 1), 0), out=buf371)
        del permute_376
        buf372 = reinterpret_tensor(buf324, (48, 128, 128), (16384, 128, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf370, (48, 128, 64), (8192, 64, 1), 0), permute_377, out=buf372)
        del permute_377
        buf374 = reinterpret_tensor(buf322, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf322  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf372, alias_58, unsqueeze_1, buf374, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_58
        buf375 = reinterpret_tensor(buf370, (48, 64, 128), (8192, 128, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_378, reinterpret_tensor(buf374, (48, 128, 128), (16384, 128, 1), 0), out=buf375)
        del permute_378
        buf376 = reinterpret_tensor(buf328, (48, 128, 64), (8192, 64, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf374, (48, 128, 128), (16384, 128, 1), 0), permute_379, out=buf376)
        del permute_379
        buf377 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf371, buf377, 393216, grid=grid(393216), stream=stream0)
        buf378 = reinterpret_tensor(buf371, (512, 768), (768, 1), 0); del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf377, permute_382, out=buf378)
        del permute_382
        buf379 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf377, (768, 512), (1, 768), 0), view_88, out=buf379)
        buf380 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf377, buf380, 3072, 128, grid=grid(3072), stream=stream0)
        buf381 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf380, buf381, 768, 4, grid=grid(768), stream=stream0)
        buf382 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf375, buf382, 512, 768, grid=grid(512, 768), stream=stream0)
        buf383 = reinterpret_tensor(buf375, (512, 768), (768, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf382, permute_387, out=buf383)
        del permute_387
        buf384 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (768, 512), (1, 768), 0), view_88, out=buf384)
        buf385 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf382, buf385, 3072, 128, grid=grid(3072), stream=stream0)
        buf386 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf385, buf386, 768, 4, grid=grid(768), stream=stream0)
        buf387 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf376, buf387, 393216, grid=grid(393216), stream=stream0)
        buf388 = reinterpret_tensor(buf376, (512, 768), (768, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf387, permute_392, out=buf388)
        del permute_392
        buf389 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (768, 512), (1, 768), 0), view_88, out=buf389)
        del view_88
        buf390 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf387, buf390, 3072, 128, grid=grid(3072), stream=stream0)
        buf391 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf390, buf391, 768, 4, grid=grid(768), stream=stream0)
        buf392 = reinterpret_tensor(buf390, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf390  # reuse
        buf396 = buf359; del buf359  # reuse
        # Source Nodes: [add_26], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf378, buf383, buf388, sqrt_8, sub_12, buf392, buf396, 3072, 128, grid=grid(3072), stream=stream0)
        buf393 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf392, buf393, 768, 4, grid=grid(768), stream=stream0)
        buf399 = buf365; del buf365  # reuse
        # Source Nodes: [add_26, mul_8, truediv_12], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf399, buf378, buf383, buf388, primals_17, sub_12, sqrt_8, 512, 768, grid=grid(512), stream=stream0)
        del primals_17
        del sqrt_8
        del sub_12
        buf397 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_26], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf396, buf397, 768, 4, grid=grid(768), stream=stream0)
        buf400 = reinterpret_tensor(buf354, (512, 3072), (3072, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (512, 768), (768, 1), 0), permute_396, out=buf400)
        del permute_396
        buf401 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (768, 512), (1, 768), 0), view_86, out=buf401)
        del view_86
        buf402 = reinterpret_tensor(buf396, (1, 768, 4), (3072, 1, 768), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf399, buf402, 3072, 128, grid=grid(3072), stream=stream0)
        buf403 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf402, buf403, 768, 4, grid=grid(768), stream=stream0)
        buf404 = reinterpret_tensor(buf400, (4, 128, 3072), (393216, 3072, 1), 0); del buf400  # reuse
        # Source Nodes: [l__mod___transformer_blocks_3_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf404, addmm_22, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_22
        buf405 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (512, 3072), (3072, 1), 0), permute_400, out=buf405)
        del permute_400
        buf406 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (3072, 512), (1, 3072), 0), view_84, out=buf406)
        del view_84
        buf407 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf404, buf407, 12288, 128, grid=grid(12288), stream=stream0)
        buf408 = reinterpret_tensor(buf402, (1, 3072), (3072, 1), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf407, buf408, 3072, 4, grid=grid(3072), stream=stream0)
        buf409 = buf392; del buf392  # reuse
        buf412 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf405, sqrt_7, sub_11, buf409, buf412, 3072, 128, grid=grid(3072), stream=stream0)
        buf410 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf409, buf410, 768, 4, grid=grid(768), stream=stream0)
        buf415 = buf399; del buf399  # reuse
        # Source Nodes: [add_23, mul_7, truediv_11], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf415, buf405, primals_15, sub_11, sqrt_7, 512, 768, grid=grid(512), stream=stream0)
        del primals_15
        del sqrt_7
        del sub_11
        buf413 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf412, buf413, 768, 4, grid=grid(768), stream=stream0)
        buf416 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (512, 768), (768, 1), 0), permute_404, out=buf416)
        del permute_404
        buf417 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (768, 512), (1, 768), 0), view_82, out=buf417)
        del view_82
        buf418 = reinterpret_tensor(buf412, (1, 768, 4), (3072, 1, 768), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf415, buf418, 3072, 128, grid=grid(3072), stream=stream0)
        buf419 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf418, buf419, 768, 4, grid=grid(768), stream=stream0)
        buf420 = reinterpret_tensor(buf383, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf416, buf420, 393216, grid=grid(393216), stream=stream0)
        buf421 = reinterpret_tensor(buf416, (48, 128, 64), (8192, 64, 1), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_409, reinterpret_tensor(buf420, (48, 128, 64), (8192, 64, 1), 0), out=buf421)
        del permute_409
        buf422 = reinterpret_tensor(buf374, (48, 128, 128), (16384, 128, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf420, (48, 128, 64), (8192, 64, 1), 0), permute_410, out=buf422)
        del permute_410
        buf424 = reinterpret_tensor(buf372, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf372  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf422, alias_61, unsqueeze_1, buf424, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_61
        buf425 = reinterpret_tensor(buf420, (48, 64, 128), (8192, 128, 1), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_411, reinterpret_tensor(buf424, (48, 128, 128), (16384, 128, 1), 0), out=buf425)
        del permute_411
        buf426 = reinterpret_tensor(buf378, (48, 128, 64), (8192, 64, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf424, (48, 128, 128), (16384, 128, 1), 0), permute_412, out=buf426)
        del permute_412
        buf427 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf421, buf427, 393216, grid=grid(393216), stream=stream0)
        buf428 = reinterpret_tensor(buf421, (512, 768), (768, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf427, permute_415, out=buf428)
        del permute_415
        buf429 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf427, (768, 512), (1, 768), 0), view_66, out=buf429)
        buf430 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf427, buf430, 3072, 128, grid=grid(3072), stream=stream0)
        buf431 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf430, buf431, 768, 4, grid=grid(768), stream=stream0)
        buf432 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf425, buf432, 512, 768, grid=grid(512, 768), stream=stream0)
        buf433 = reinterpret_tensor(buf425, (512, 768), (768, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf432, permute_420, out=buf433)
        del permute_420
        buf434 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (768, 512), (1, 768), 0), view_66, out=buf434)
        buf435 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf432, buf435, 3072, 128, grid=grid(3072), stream=stream0)
        buf436 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf435, buf436, 768, 4, grid=grid(768), stream=stream0)
        buf437 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf426, buf437, 393216, grid=grid(393216), stream=stream0)
        buf438 = reinterpret_tensor(buf426, (512, 768), (768, 1), 0); del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf437, permute_425, out=buf438)
        del permute_425
        buf439 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (768, 512), (1, 768), 0), view_66, out=buf439)
        del view_66
        buf440 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf437, buf440, 3072, 128, grid=grid(3072), stream=stream0)
        buf441 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf440, buf441, 768, 4, grid=grid(768), stream=stream0)
        buf442 = reinterpret_tensor(buf440, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf440  # reuse
        buf446 = buf409; del buf409  # reuse
        # Source Nodes: [add_20], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf428, buf433, buf438, sqrt_6, sub_9, buf442, buf446, 3072, 128, grid=grid(3072), stream=stream0)
        buf443 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf442, buf443, 768, 4, grid=grid(768), stream=stream0)
        buf449 = buf415; del buf415  # reuse
        # Source Nodes: [add_20, mul_6, truediv_9], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf449, buf428, buf433, buf438, primals_13, sub_9, sqrt_6, 512, 768, grid=grid(512), stream=stream0)
        del primals_13
        del sqrt_6
        del sub_9
        buf447 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf446, buf447, 768, 4, grid=grid(768), stream=stream0)
        buf450 = reinterpret_tensor(buf404, (512, 3072), (3072, 1), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (512, 768), (768, 1), 0), permute_429, out=buf450)
        del permute_429
        buf451 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (768, 512), (1, 768), 0), view_64, out=buf451)
        del view_64
        buf452 = reinterpret_tensor(buf446, (1, 768, 4), (3072, 1, 768), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf449, buf452, 3072, 128, grid=grid(3072), stream=stream0)
        buf453 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf452, buf453, 768, 4, grid=grid(768), stream=stream0)
        buf454 = reinterpret_tensor(buf450, (4, 128, 3072), (393216, 3072, 1), 0); del buf450  # reuse
        # Source Nodes: [l__mod___transformer_blocks_2_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf454, addmm_16, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_16
        buf455 = buf438; del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (512, 3072), (3072, 1), 0), permute_433, out=buf455)
        del permute_433
        buf456 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (3072, 512), (1, 3072), 0), view_62, out=buf456)
        del view_62
        buf457 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf454, buf457, 12288, 128, grid=grid(12288), stream=stream0)
        buf458 = reinterpret_tensor(buf452, (1, 3072), (3072, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf457, buf458, 3072, 4, grid=grid(3072), stream=stream0)
        buf459 = buf442; del buf442  # reuse
        buf462 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf455, sqrt_5, sub_8, buf459, buf462, 3072, 128, grid=grid(3072), stream=stream0)
        buf460 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf459, buf460, 768, 4, grid=grid(768), stream=stream0)
        buf465 = buf449; del buf449  # reuse
        # Source Nodes: [add_17, mul_5, truediv_8], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf465, buf455, primals_11, sub_8, sqrt_5, 512, 768, grid=grid(512), stream=stream0)
        del primals_11
        del sqrt_5
        del sub_8
        buf463 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf462, buf463, 768, 4, grid=grid(768), stream=stream0)
        buf466 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (512, 768), (768, 1), 0), permute_437, out=buf466)
        del permute_437
        buf467 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (768, 512), (1, 768), 0), view_60, out=buf467)
        del view_60
        buf468 = reinterpret_tensor(buf462, (1, 768, 4), (3072, 1, 768), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf465, buf468, 3072, 128, grid=grid(3072), stream=stream0)
        buf469 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf468, buf469, 768, 4, grid=grid(768), stream=stream0)
        buf470 = reinterpret_tensor(buf433, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf466, buf470, 393216, grid=grid(393216), stream=stream0)
        buf471 = reinterpret_tensor(buf466, (48, 128, 64), (8192, 64, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_442, reinterpret_tensor(buf470, (48, 128, 64), (8192, 64, 1), 0), out=buf471)
        del permute_442
        buf472 = reinterpret_tensor(buf424, (48, 128, 128), (16384, 128, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf470, (48, 128, 64), (8192, 64, 1), 0), permute_443, out=buf472)
        del permute_443
        buf474 = reinterpret_tensor(buf422, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf422  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf472, alias_64, unsqueeze_1, buf474, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_64
        buf475 = reinterpret_tensor(buf470, (48, 64, 128), (8192, 128, 1), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_444, reinterpret_tensor(buf474, (48, 128, 128), (16384, 128, 1), 0), out=buf475)
        del permute_444
        buf476 = reinterpret_tensor(buf428, (48, 128, 64), (8192, 64, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf474, (48, 128, 128), (16384, 128, 1), 0), permute_445, out=buf476)
        del permute_445
        buf477 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf471, buf477, 393216, grid=grid(393216), stream=stream0)
        buf478 = reinterpret_tensor(buf471, (512, 768), (768, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf477, permute_448, out=buf478)
        del permute_448
        buf479 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf477, (768, 512), (1, 768), 0), view_44, out=buf479)
        buf480 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf477, buf480, 3072, 128, grid=grid(3072), stream=stream0)
        buf481 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf480, buf481, 768, 4, grid=grid(768), stream=stream0)
        buf482 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf475, buf482, 512, 768, grid=grid(512, 768), stream=stream0)
        buf483 = reinterpret_tensor(buf475, (512, 768), (768, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf482, permute_453, out=buf483)
        del permute_453
        buf484 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (768, 512), (1, 768), 0), view_44, out=buf484)
        buf485 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf482, buf485, 3072, 128, grid=grid(3072), stream=stream0)
        buf486 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf485, buf486, 768, 4, grid=grid(768), stream=stream0)
        buf487 = buf482; del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf476, buf487, 393216, grid=grid(393216), stream=stream0)
        buf488 = reinterpret_tensor(buf476, (512, 768), (768, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf487, permute_458, out=buf488)
        del permute_458
        buf489 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf487, (768, 512), (1, 768), 0), view_44, out=buf489)
        del view_44
        buf490 = buf485; del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf487, buf490, 3072, 128, grid=grid(3072), stream=stream0)
        buf491 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf490, buf491, 768, 4, grid=grid(768), stream=stream0)
        buf492 = reinterpret_tensor(buf490, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf490  # reuse
        buf496 = buf459; del buf459  # reuse
        # Source Nodes: [add_14], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf478, buf483, buf488, sqrt_4, sub_6, buf492, buf496, 3072, 128, grid=grid(3072), stream=stream0)
        buf493 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf492, buf493, 768, 4, grid=grid(768), stream=stream0)
        buf499 = buf465; del buf465  # reuse
        # Source Nodes: [add_14, mul_4, truediv_6], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf499, buf478, buf483, buf488, primals_9, sub_6, sqrt_4, 512, 768, grid=grid(512), stream=stream0)
        del primals_9
        del sqrt_4
        del sub_6
        buf497 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf496, buf497, 768, 4, grid=grid(768), stream=stream0)
        buf500 = reinterpret_tensor(buf454, (512, 3072), (3072, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (512, 768), (768, 1), 0), permute_462, out=buf500)
        del permute_462
        buf501 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (768, 512), (1, 768), 0), view_42, out=buf501)
        del view_42
        buf502 = reinterpret_tensor(buf496, (1, 768, 4), (3072, 1, 768), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf499, buf502, 3072, 128, grid=grid(3072), stream=stream0)
        buf503 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf502, buf503, 768, 4, grid=grid(768), stream=stream0)
        buf504 = reinterpret_tensor(buf500, (4, 128, 3072), (393216, 3072, 1), 0); del buf500  # reuse
        # Source Nodes: [l__mod___transformer_blocks_1_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf504, addmm_10, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_10
        buf505 = buf488; del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (512, 3072), (3072, 1), 0), permute_466, out=buf505)
        del permute_466
        buf506 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (3072, 512), (1, 3072), 0), view_40, out=buf506)
        del view_40
        buf507 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf504, buf507, 12288, 128, grid=grid(12288), stream=stream0)
        buf508 = reinterpret_tensor(buf502, (1, 3072), (3072, 1), 0); del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf507, buf508, 3072, 4, grid=grid(3072), stream=stream0)
        buf509 = buf492; del buf492  # reuse
        buf512 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf505, sqrt_3, sub_5, buf509, buf512, 3072, 128, grid=grid(3072), stream=stream0)
        buf510 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf509, buf510, 768, 4, grid=grid(768), stream=stream0)
        buf515 = buf499; del buf499  # reuse
        # Source Nodes: [add_11, mul_3, truediv_5], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf515, buf505, primals_7, sub_5, sqrt_3, 512, 768, grid=grid(512), stream=stream0)
        del primals_7
        del sqrt_3
        del sub_5
        buf513 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf512, buf513, 768, 4, grid=grid(768), stream=stream0)
        buf516 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf515, (512, 768), (768, 1), 0), permute_470, out=buf516)
        del permute_470
        buf517 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf515, (768, 512), (1, 768), 0), view_38, out=buf517)
        del view_38
        buf518 = reinterpret_tensor(buf512, (1, 768, 4), (3072, 1, 768), 0); del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf515, buf518, 3072, 128, grid=grid(3072), stream=stream0)
        buf519 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf518, buf519, 768, 4, grid=grid(768), stream=stream0)
        buf520 = reinterpret_tensor(buf483, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf516, buf520, 393216, grid=grid(393216), stream=stream0)
        buf521 = reinterpret_tensor(buf516, (48, 128, 64), (8192, 64, 1), 0); del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_475, reinterpret_tensor(buf520, (48, 128, 64), (8192, 64, 1), 0), out=buf521)
        del permute_475
        buf522 = reinterpret_tensor(buf474, (48, 128, 128), (16384, 128, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf520, (48, 128, 64), (8192, 64, 1), 0), permute_476, out=buf522)
        del permute_476
        buf524 = reinterpret_tensor(buf472, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf472  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf522, alias_67, unsqueeze_1, buf524, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_67
        buf525 = reinterpret_tensor(buf520, (48, 64, 128), (8192, 128, 1), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_477, reinterpret_tensor(buf524, (48, 128, 128), (16384, 128, 1), 0), out=buf525)
        del permute_477
        buf526 = reinterpret_tensor(buf478, (48, 128, 64), (8192, 64, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf524, (48, 128, 128), (16384, 128, 1), 0), permute_478, out=buf526)
        del permute_478
        buf527 = buf487; del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf521, buf527, 393216, grid=grid(393216), stream=stream0)
        buf528 = reinterpret_tensor(buf521, (512, 768), (768, 1), 0); del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf527, permute_481, out=buf528)
        del permute_481
        buf529 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (768, 512), (1, 768), 0), view_22, out=buf529)
        buf530 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf527, buf530, 3072, 128, grid=grid(3072), stream=stream0)
        buf531 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf530, buf531, 768, 4, grid=grid(768), stream=stream0)
        buf532 = buf527; del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf525, buf532, 512, 768, grid=grid(512, 768), stream=stream0)
        buf533 = reinterpret_tensor(buf525, (512, 768), (768, 1), 0); del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf532, permute_486, out=buf533)
        del permute_486
        buf534 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (768, 512), (1, 768), 0), view_22, out=buf534)
        buf535 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf532, buf535, 3072, 128, grid=grid(3072), stream=stream0)
        buf536 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf535, buf536, 768, 4, grid=grid(768), stream=stream0)
        buf537 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf526, buf537, 393216, grid=grid(393216), stream=stream0)
        buf538 = reinterpret_tensor(buf526, (512, 768), (768, 1), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf537, permute_491, out=buf538)
        del permute_491
        buf539 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf537, (768, 512), (1, 768), 0), view_22, out=buf539)
        del view_22
        buf540 = buf535; del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf537, buf540, 3072, 128, grid=grid(3072), stream=stream0)
        buf541 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf540, buf541, 768, 4, grid=grid(768), stream=stream0)
        buf542 = reinterpret_tensor(buf540, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf540  # reuse
        buf546 = buf509; del buf509  # reuse
        # Source Nodes: [add_8], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf528, buf533, buf538, sqrt_2, sub_3, buf542, buf546, 3072, 128, grid=grid(3072), stream=stream0)
        buf543 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf542, buf543, 768, 4, grid=grid(768), stream=stream0)
        buf549 = buf515; del buf515  # reuse
        # Source Nodes: [add_8, mul_2, truediv_3], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_13.run(buf549, buf528, buf533, buf538, primals_5, sub_3, sqrt_2, 512, 768, grid=grid(512), stream=stream0)
        del primals_5
        del sqrt_2
        del sub_3
        buf547 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf546, buf547, 768, 4, grid=grid(768), stream=stream0)
        buf550 = reinterpret_tensor(buf504, (512, 3072), (3072, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (512, 768), (768, 1), 0), permute_495, out=buf550)
        del permute_495
        buf551 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (768, 512), (1, 768), 0), view_20, out=buf551)
        del view_20
        buf552 = reinterpret_tensor(buf546, (1, 768, 4), (3072, 1, 768), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf549, buf552, 3072, 128, grid=grid(3072), stream=stream0)
        buf553 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf552, buf553, 768, 4, grid=grid(768), stream=stream0)
        buf554 = reinterpret_tensor(buf550, (4, 128, 3072), (393216, 3072, 1), 0); del buf550  # reuse
        # Source Nodes: [l__mod___transformer_blocks_0_feed_forward_activation], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf554, addmm_4, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_4
        buf555 = buf538; del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf554, (512, 3072), (3072, 1), 0), permute_499, out=buf555)
        del permute_499
        buf556 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf554, (3072, 512), (1, 3072), 0), view_18, out=buf556)
        del view_18
        buf557 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf554, buf557, 12288, 128, grid=grid(12288), stream=stream0)
        del buf554
        buf558 = reinterpret_tensor(buf552, (1, 3072), (3072, 1), 0); del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf557, buf558, 3072, 4, grid=grid(3072), stream=stream0)
        del buf557
        buf559 = buf542; del buf542  # reuse
        buf562 = empty_strided((1, 1, 768, 4), (3072, 3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_5.run(buf555, sqrt_1, sub_2, buf559, buf562, 3072, 128, grid=grid(3072), stream=stream0)
        buf560 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf559, buf560, 768, 4, grid=grid(768), stream=stream0)
        buf565 = buf549; del buf549  # reuse
        # Source Nodes: [add_5, mul_1, truediv_2], Original ATen: [aten.add, aten.div, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_eq_masked_fill_mul_neg_sum_14.run(buf565, buf555, primals_3, sub_2, sqrt_1, 512, 768, grid=grid(512), stream=stream0)
        del primals_3
        del sqrt_1
        del sub_2
        buf563 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf562, buf563, 768, 4, grid=grid(768), stream=stream0)
        buf566 = buf555; del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (512, 768), (768, 1), 0), permute_503, out=buf566)
        del permute_503
        buf567 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (768, 512), (1, 768), 0), view_16, out=buf567)
        del view_16
        buf568 = reinterpret_tensor(buf562, (1, 768, 4), (3072, 1, 768), 0); del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_0.run(buf565, buf568, 3072, 128, grid=grid(3072), stream=stream0)
        buf569 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf568, buf569, 768, 4, grid=grid(768), stream=stream0)
        buf570 = reinterpret_tensor(buf533, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf566, buf570, 393216, grid=grid(393216), stream=stream0)
        buf571 = reinterpret_tensor(buf566, (48, 128, 64), (8192, 64, 1), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_508, reinterpret_tensor(buf570, (48, 128, 64), (8192, 64, 1), 0), out=buf571)
        del permute_508
        buf572 = reinterpret_tensor(buf524, (48, 128, 128), (16384, 128, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf570, (48, 128, 64), (8192, 64, 1), 0), permute_509, out=buf572)
        del permute_509
        buf574 = reinterpret_tensor(buf522, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf522  # reuse
        # Source Nodes: [eq], Original ATen: [aten._softmax_backward_data, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_backward_data_div_eq_masked_fill_8.run(buf572, alias_70, unsqueeze_1, buf574, 6144, 128, grid=grid(6144), stream=stream0)
        del alias_70
        del buf572
        del unsqueeze_1
        buf575 = reinterpret_tensor(buf570, (48, 64, 128), (8192, 128, 1), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_510, reinterpret_tensor(buf574, (48, 128, 128), (16384, 128, 1), 0), out=buf575)
        del permute_510
        buf576 = reinterpret_tensor(buf528, (48, 128, 64), (8192, 64, 1), 0); del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf574, (48, 128, 128), (16384, 128, 1), 0), permute_511, out=buf576)
        del buf574
        del permute_511
        buf577 = buf537; del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf571, buf577, 393216, grid=grid(393216), stream=stream0)
        buf578 = reinterpret_tensor(buf571, (512, 768), (768, 1), 0); del buf571  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf577, permute_514, out=buf578)
        del permute_514
        buf579 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (768, 512), (1, 768), 0), view, out=buf579)
        buf580 = buf568; del buf568  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf577, buf580, 3072, 128, grid=grid(3072), stream=stream0)
        buf581 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf580, buf581, 768, 4, grid=grid(768), stream=stream0)
        buf582 = buf577; del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf575, buf582, 512, 768, grid=grid(512, 768), stream=stream0)
        buf583 = reinterpret_tensor(buf575, (512, 768), (768, 1), 0); del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf582, permute_519, out=buf583)
        del permute_519
        buf584 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf582, (768, 512), (1, 768), 0), view, out=buf584)
        buf585 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf582, buf585, 3072, 128, grid=grid(3072), stream=stream0)
        buf586 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf585, buf586, 768, 4, grid=grid(768), stream=stream0)
        buf587 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(buf576, buf587, 393216, grid=grid(393216), stream=stream0)
        buf588 = reinterpret_tensor(buf576, (512, 768), (768, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf587, permute_524, out=buf588)
        del permute_524
        buf589 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf587, (768, 512), (1, 768), 0), view, out=buf589)
        del view
        buf590 = buf585; del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf587, buf590, 3072, 128, grid=grid(3072), stream=stream0)
        buf591 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf590, buf591, 768, 4, grid=grid(768), stream=stream0)
        buf592 = reinterpret_tensor(buf590, (1, 1, 768, 4), (3072, 3072, 1, 768), 0); del buf590  # reuse
        buf596 = buf559; del buf559  # reuse
        # Source Nodes: [add_2], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_12.run(buf578, buf583, buf588, sqrt, sub, buf592, buf596, 3072, 128, grid=grid(3072), stream=stream0)
        buf593 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_sum_1.run(buf592, buf593, 768, 4, grid=grid(768), stream=stream0)
        del buf592
        buf599 = buf565; del buf565  # reuse
        buf601 = reinterpret_tensor(buf587, (4, 128, 768), (98304, 768, 1), 0); del buf587  # reuse
        buf605 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, mul, truediv], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.eq, aten.masked_fill, aten.mul, aten.neg, aten.sum]
        triton_per_fused_add_div_embedding_dense_backward_eq_masked_fill_mul_neg_sum_15.run(buf599, buf578, buf583, buf588, primals_1, sub, sqrt, primals_197, primals_196, buf601, buf605, 512, 768, grid=grid(512), stream=stream0)
        del buf578
        del buf583
        del buf588
        del buf599
        del primals_1
        del sqrt
        del sub
        buf597 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_sum_1.run(buf596, buf597, 768, 4, grid=grid(768), stream=stream0)
        del buf596
        buf600 = empty((3, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf600, 2304, grid=grid(2304), stream=stream0)
        aten.index_put_(buf600, [primals_197], buf601, True)
        del buf601
        del primals_197
        buf604 = empty((20005, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_17.run(buf604, 15363840, grid=grid(15363840), stream=stream0)
        aten.index_put_(buf604, [primals_196], buf605, True)
        del buf605
        del primals_196
        return (reinterpret_tensor(buf597, (768, ), (1, ), 0), reinterpret_tensor(buf593, (768, ), (1, ), 0), reinterpret_tensor(buf563, (768, ), (1, ), 0), reinterpret_tensor(buf560, (768, ), (1, ), 0), reinterpret_tensor(buf547, (768, ), (1, ), 0), reinterpret_tensor(buf543, (768, ), (1, ), 0), reinterpret_tensor(buf513, (768, ), (1, ), 0), reinterpret_tensor(buf510, (768, ), (1, ), 0), reinterpret_tensor(buf497, (768, ), (1, ), 0), reinterpret_tensor(buf493, (768, ), (1, ), 0), reinterpret_tensor(buf463, (768, ), (1, ), 0), reinterpret_tensor(buf460, (768, ), (1, ), 0), reinterpret_tensor(buf447, (768, ), (1, ), 0), reinterpret_tensor(buf443, (768, ), (1, ), 0), reinterpret_tensor(buf413, (768, ), (1, ), 0), reinterpret_tensor(buf410, (768, ), (1, ), 0), reinterpret_tensor(buf397, (768, ), (1, ), 0), reinterpret_tensor(buf393, (768, ), (1, ), 0), reinterpret_tensor(buf363, (768, ), (1, ), 0), reinterpret_tensor(buf360, (768, ), (1, ), 0), reinterpret_tensor(buf347, (768, ), (1, ), 0), reinterpret_tensor(buf343, (768, ), (1, ), 0), reinterpret_tensor(buf313, (768, ), (1, ), 0), reinterpret_tensor(buf310, (768, ), (1, ), 0), reinterpret_tensor(buf297, (768, ), (1, ), 0), reinterpret_tensor(buf293, (768, ), (1, ), 0), reinterpret_tensor(buf263, (768, ), (1, ), 0), reinterpret_tensor(buf260, (768, ), (1, ), 0), reinterpret_tensor(buf247, (768, ), (1, ), 0), reinterpret_tensor(buf243, (768, ), (1, ), 0), reinterpret_tensor(buf213, (768, ), (1, ), 0), reinterpret_tensor(buf210, (768, ), (1, ), 0), reinterpret_tensor(buf197, (768, ), (1, ), 0), reinterpret_tensor(buf193, (768, ), (1, ), 0), reinterpret_tensor(buf163, (768, ), (1, ), 0), reinterpret_tensor(buf160, (768, ), (1, ), 0), reinterpret_tensor(buf147, (768, ), (1, ), 0), reinterpret_tensor(buf143, (768, ), (1, ), 0), reinterpret_tensor(buf113, (768, ), (1, ), 0), reinterpret_tensor(buf110, (768, ), (1, ), 0), reinterpret_tensor(buf97, (768, ), (1, ), 0), reinterpret_tensor(buf93, (768, ), (1, ), 0), reinterpret_tensor(buf63, (768, ), (1, ), 0), reinterpret_tensor(buf60, (768, ), (1, ), 0), reinterpret_tensor(buf47, (768, ), (1, ), 0), reinterpret_tensor(buf43, (768, ), (1, ), 0), reinterpret_tensor(buf13, (768, ), (1, ), 0), reinterpret_tensor(buf10, (768, ), (1, ), 0), buf604, buf600, reinterpret_tensor(buf589, (768, 768), (768, 1), 0), reinterpret_tensor(buf591, (768, ), (1, ), 0), reinterpret_tensor(buf584, (768, 768), (768, 1), 0), reinterpret_tensor(buf586, (768, ), (1, ), 0), reinterpret_tensor(buf579, (768, 768), (768, 1), 0), reinterpret_tensor(buf581, (768, ), (1, ), 0), reinterpret_tensor(buf567, (768, 768), (768, 1), 0), reinterpret_tensor(buf569, (768, ), (1, ), 0), reinterpret_tensor(buf556, (3072, 768), (768, 1), 0), reinterpret_tensor(buf558, (3072, ), (1, ), 0), reinterpret_tensor(buf551, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf553, (768, ), (1, ), 0), reinterpret_tensor(buf539, (768, 768), (768, 1), 0), reinterpret_tensor(buf541, (768, ), (1, ), 0), reinterpret_tensor(buf534, (768, 768), (768, 1), 0), reinterpret_tensor(buf536, (768, ), (1, ), 0), reinterpret_tensor(buf529, (768, 768), (768, 1), 0), reinterpret_tensor(buf531, (768, ), (1, ), 0), reinterpret_tensor(buf517, (768, 768), (768, 1), 0), reinterpret_tensor(buf519, (768, ), (1, ), 0), reinterpret_tensor(buf506, (3072, 768), (768, 1), 0), reinterpret_tensor(buf508, (3072, ), (1, ), 0), reinterpret_tensor(buf501, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf503, (768, ), (1, ), 0), reinterpret_tensor(buf489, (768, 768), (768, 1), 0), reinterpret_tensor(buf491, (768, ), (1, ), 0), reinterpret_tensor(buf484, (768, 768), (768, 1), 0), reinterpret_tensor(buf486, (768, ), (1, ), 0), reinterpret_tensor(buf479, (768, 768), (768, 1), 0), reinterpret_tensor(buf481, (768, ), (1, ), 0), reinterpret_tensor(buf467, (768, 768), (768, 1), 0), reinterpret_tensor(buf469, (768, ), (1, ), 0), reinterpret_tensor(buf456, (3072, 768), (768, 1), 0), reinterpret_tensor(buf458, (3072, ), (1, ), 0), reinterpret_tensor(buf451, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf453, (768, ), (1, ), 0), reinterpret_tensor(buf439, (768, 768), (768, 1), 0), reinterpret_tensor(buf441, (768, ), (1, ), 0), reinterpret_tensor(buf434, (768, 768), (768, 1), 0), reinterpret_tensor(buf436, (768, ), (1, ), 0), reinterpret_tensor(buf429, (768, 768), (768, 1), 0), reinterpret_tensor(buf431, (768, ), (1, ), 0), reinterpret_tensor(buf417, (768, 768), (768, 1), 0), reinterpret_tensor(buf419, (768, ), (1, ), 0), reinterpret_tensor(buf406, (3072, 768), (768, 1), 0), reinterpret_tensor(buf408, (3072, ), (1, ), 0), reinterpret_tensor(buf401, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf403, (768, ), (1, ), 0), reinterpret_tensor(buf389, (768, 768), (768, 1), 0), reinterpret_tensor(buf391, (768, ), (1, ), 0), reinterpret_tensor(buf384, (768, 768), (768, 1), 0), reinterpret_tensor(buf386, (768, ), (1, ), 0), reinterpret_tensor(buf379, (768, 768), (768, 1), 0), reinterpret_tensor(buf381, (768, ), (1, ), 0), reinterpret_tensor(buf367, (768, 768), (768, 1), 0), reinterpret_tensor(buf369, (768, ), (1, ), 0), reinterpret_tensor(buf356, (3072, 768), (768, 1), 0), reinterpret_tensor(buf358, (3072, ), (1, ), 0), reinterpret_tensor(buf351, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf353, (768, ), (1, ), 0), reinterpret_tensor(buf339, (768, 768), (768, 1), 0), reinterpret_tensor(buf341, (768, ), (1, ), 0), reinterpret_tensor(buf334, (768, 768), (768, 1), 0), reinterpret_tensor(buf336, (768, ), (1, ), 0), reinterpret_tensor(buf329, (768, 768), (768, 1), 0), reinterpret_tensor(buf331, (768, ), (1, ), 0), reinterpret_tensor(buf317, (768, 768), (768, 1), 0), reinterpret_tensor(buf319, (768, ), (1, ), 0), reinterpret_tensor(buf306, (3072, 768), (768, 1), 0), reinterpret_tensor(buf308, (3072, ), (1, ), 0), reinterpret_tensor(buf301, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf303, (768, ), (1, ), 0), reinterpret_tensor(buf289, (768, 768), (768, 1), 0), reinterpret_tensor(buf291, (768, ), (1, ), 0), reinterpret_tensor(buf284, (768, 768), (768, 1), 0), reinterpret_tensor(buf286, (768, ), (1, ), 0), reinterpret_tensor(buf279, (768, 768), (768, 1), 0), reinterpret_tensor(buf281, (768, ), (1, ), 0), reinterpret_tensor(buf267, (768, 768), (768, 1), 0), reinterpret_tensor(buf269, (768, ), (1, ), 0), reinterpret_tensor(buf256, (3072, 768), (768, 1), 0), reinterpret_tensor(buf258, (3072, ), (1, ), 0), reinterpret_tensor(buf251, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf253, (768, ), (1, ), 0), reinterpret_tensor(buf239, (768, 768), (768, 1), 0), reinterpret_tensor(buf241, (768, ), (1, ), 0), reinterpret_tensor(buf234, (768, 768), (768, 1), 0), reinterpret_tensor(buf236, (768, ), (1, ), 0), reinterpret_tensor(buf229, (768, 768), (768, 1), 0), reinterpret_tensor(buf231, (768, ), (1, ), 0), reinterpret_tensor(buf217, (768, 768), (768, 1), 0), reinterpret_tensor(buf219, (768, ), (1, ), 0), reinterpret_tensor(buf206, (3072, 768), (768, 1), 0), reinterpret_tensor(buf208, (3072, ), (1, ), 0), reinterpret_tensor(buf201, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf203, (768, ), (1, ), 0), reinterpret_tensor(buf189, (768, 768), (768, 1), 0), reinterpret_tensor(buf191, (768, ), (1, ), 0), reinterpret_tensor(buf184, (768, 768), (768, 1), 0), reinterpret_tensor(buf186, (768, ), (1, ), 0), reinterpret_tensor(buf179, (768, 768), (768, 1), 0), reinterpret_tensor(buf181, (768, ), (1, ), 0), reinterpret_tensor(buf167, (768, 768), (768, 1), 0), reinterpret_tensor(buf169, (768, ), (1, ), 0), reinterpret_tensor(buf156, (3072, 768), (768, 1), 0), reinterpret_tensor(buf158, (3072, ), (1, ), 0), reinterpret_tensor(buf151, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf153, (768, ), (1, ), 0), reinterpret_tensor(buf139, (768, 768), (768, 1), 0), reinterpret_tensor(buf141, (768, ), (1, ), 0), reinterpret_tensor(buf134, (768, 768), (768, 1), 0), reinterpret_tensor(buf136, (768, ), (1, ), 0), reinterpret_tensor(buf129, (768, 768), (768, 1), 0), reinterpret_tensor(buf131, (768, ), (1, ), 0), reinterpret_tensor(buf117, (768, 768), (768, 1), 0), reinterpret_tensor(buf119, (768, ), (1, ), 0), reinterpret_tensor(buf106, (3072, 768), (768, 1), 0), reinterpret_tensor(buf108, (3072, ), (1, ), 0), reinterpret_tensor(buf101, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf103, (768, ), (1, ), 0), reinterpret_tensor(buf89, (768, 768), (768, 1), 0), reinterpret_tensor(buf91, (768, ), (1, ), 0), reinterpret_tensor(buf84, (768, 768), (768, 1), 0), reinterpret_tensor(buf86, (768, ), (1, ), 0), reinterpret_tensor(buf79, (768, 768), (768, 1), 0), reinterpret_tensor(buf81, (768, ), (1, ), 0), reinterpret_tensor(buf67, (768, 768), (768, 1), 0), reinterpret_tensor(buf69, (768, ), (1, ), 0), reinterpret_tensor(buf56, (3072, 768), (768, 1), 0), reinterpret_tensor(buf58, (3072, ), (1, ), 0), reinterpret_tensor(buf51, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf53, (768, ), (1, ), 0), reinterpret_tensor(buf39, (768, 768), (768, 1), 0), reinterpret_tensor(buf41, (768, ), (1, ), 0), reinterpret_tensor(buf34, (768, 768), (768, 1), 0), reinterpret_tensor(buf36, (768, ), (1, ), 0), reinterpret_tensor(buf29, (768, 768), (768, 1), 0), reinterpret_tensor(buf31, (768, ), (1, ), 0), reinterpret_tensor(buf17, (768, 768), (768, 1), 0), reinterpret_tensor(buf19, (768, ), (1, ), 0), reinterpret_tensor(buf6, (3072, 768), (768, 1), 0), reinterpret_tensor(buf8, (3072, ), (1, ), 0), reinterpret_tensor(buf1, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf3, (768, ), (1, ), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((4, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_197 = rand_strided((4, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    unsqueeze_1 = rand_strided((4, 1, 128, 128), (16384, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    sqrt = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_1 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_2 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_2 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_3 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_3 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_5 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_4 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_6 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_5 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_8 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_6 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_9 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_7 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_11 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_8 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_12 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_9 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_14 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_10 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_15 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_11 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_17 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_12 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_18 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_13 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_20 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_14 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_21 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_15 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_23 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_16 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_24 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_17 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_26 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_18 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_27 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_19 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_29 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_20 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_30 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_236 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_21 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_32 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    sqrt_22 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_33 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_258 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_23 = rand_strided((4, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_35 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_132 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_136 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_140 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_145 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_147 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_148 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_161 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_165 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_169 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_173 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_178 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_40 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_180 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_181 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_198 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_202 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_206 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_211 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_213 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_214 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_217 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_235 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_239 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_46 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_246 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_247 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_260 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_268 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_272 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_277 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_49 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_279 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_280 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_293 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_297 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_301 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_305 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_52 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_312 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_313 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_316 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_326 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_330 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_334 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_338 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_343 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_55 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_345 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_346 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_349 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_359 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_363 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_367 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_371 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_376 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_58 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_378 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_379 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_382 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_392 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_396 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_400 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_409 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_61 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_411 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_412 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_420 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_425 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_429 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_433 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_437 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_442 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_64 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_444 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_445 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_458 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_462 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_466 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_470 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_67 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_477 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_478 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_481 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_491 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_495 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_499 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_503 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_508 = rand_strided((48, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_70 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_510 = rand_strided((48, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_511 = rand_strided((48, 128, 64), (8192, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_514 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_519 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_524 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_196, primals_197, unsqueeze_1, sqrt, sub, view, view_16, sqrt_1, sub_2, view_18, addmm_4, view_20, sqrt_2, sub_3, view_22, view_38, sqrt_3, sub_5, view_40, addmm_10, view_42, sqrt_4, sub_6, view_44, view_60, sqrt_5, sub_8, view_62, addmm_16, view_64, sqrt_6, sub_9, view_66, view_82, sqrt_7, sub_11, view_84, addmm_22, view_86, sqrt_8, sub_12, view_88, view_104, sqrt_9, sub_14, view_106, addmm_28, view_108, sqrt_10, sub_15, view_110, view_126, sqrt_11, sub_17, view_128, addmm_34, view_130, sqrt_12, sub_18, view_132, view_148, sqrt_13, sub_20, view_150, addmm_40, view_152, sqrt_14, sub_21, view_154, view_170, sqrt_15, sub_23, view_172, addmm_46, view_174, sqrt_16, sub_24, view_176, view_192, sqrt_17, sub_26, view_194, addmm_52, view_196, sqrt_18, sub_27, view_198, view_214, sqrt_19, sub_29, view_216, addmm_58, view_218, sqrt_20, sub_30, view_220, view_236, sqrt_21, sub_32, view_238, addmm_64, view_240, sqrt_22, sub_33, view_242, view_258, sqrt_23, sub_35, view_260, addmm_70, view_262, permute_132, permute_136, permute_140, permute_145, permute_146, alias_37, permute_147, permute_148, permute_151, permute_156, permute_161, permute_165, permute_169, permute_173, permute_178, permute_179, alias_40, permute_180, permute_181, permute_184, permute_189, permute_194, permute_198, permute_202, permute_206, permute_211, permute_212, alias_43, permute_213, permute_214, permute_217, permute_222, permute_227, permute_231, permute_235, permute_239, permute_244, permute_245, alias_46, permute_246, permute_247, permute_250, permute_255, permute_260, permute_264, permute_268, permute_272, permute_277, permute_278, alias_49, permute_279, permute_280, permute_283, permute_288, permute_293, permute_297, permute_301, permute_305, permute_310, permute_311, alias_52, permute_312, permute_313, permute_316, permute_321, permute_326, permute_330, permute_334, permute_338, permute_343, permute_344, alias_55, permute_345, permute_346, permute_349, permute_354, permute_359, permute_363, permute_367, permute_371, permute_376, permute_377, alias_58, permute_378, permute_379, permute_382, permute_387, permute_392, permute_396, permute_400, permute_404, permute_409, permute_410, alias_61, permute_411, permute_412, permute_415, permute_420, permute_425, permute_429, permute_433, permute_437, permute_442, permute_443, alias_64, permute_444, permute_445, permute_448, permute_453, permute_458, permute_462, permute_466, permute_470, permute_475, permute_476, alias_67, permute_477, permute_478, permute_481, permute_486, permute_491, permute_495, permute_499, permute_503, permute_508, permute_509, alias_70, permute_510, permute_511, permute_514, permute_519, permute_524, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BERT_pytorch', benchmark_compiled_module)
