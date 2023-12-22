
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


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmwed34vggqrgigdymugrotk55rcnnws46spo4jaa6rpsekfg3j.py
# Source Nodes: [], Original ATen: [aten.nll_loss_backward]

triton_poi_fused_nll_loss_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/us/cusiultoolk22d4v7olgokbbg4khlba426bio7pnsmccbbyh2yga.py
# Source Nodes: [end_loss, query_states, start_loss], Original ATen: [aten._log_softmax_backward_data, aten.div, aten.masked_fill, aten.nll_loss_backward, aten.nll_loss_forward]
# end_loss => convert_element_type_50, sum_17
# query_states => full_default_1
# start_loss => convert_element_type_49, sum_14
triton_per_fused__log_softmax_backward_data_div_masked_fill_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*i1', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_div_masked_fill_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (0)).to(tl.int1)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = tl.load(in_ptr3 + (0)).to(tl.int1)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp19 = tl.load(in_ptr4 + (r0), rmask, other=0.0)
    tmp20 = tl.load(in_ptr5 + (0)).to(tl.int1)
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp22 = tl.load(in_ptr6 + (0)).to(tl.int1)
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp5 = 2.0
    tmp6 = tmp4 / tmp5
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp6 / tmp10
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp24 = tmp23.to(tl.int64)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp6 / tmp25
    tmp27 = tl.where(tmp21, tmp26, tmp12)
    tmp28 = tmp19 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp18, None)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgj6khi67bfhjval4fmhayskx3e7twgobce3ccetenhdqq55uabw.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*i1', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = (xindex // 2)
    x2 = xindex
    tmp7 = tl.load(in_ptr2 + (0)).to(tl.int1)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp13 = tl.load(in_ptr4 + (0)).to(tl.int1)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp23 = tl.load(in_ptr6 + (0))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp35 = tl.load(in_ptr9 + (0)).to(tl.int1)
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp37 = tl.load(in_ptr10 + (0)).to(tl.int1)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp46 = tl.load(in_ptr12 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = 2.0
    tmp12 = tmp10 / tmp11
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp12 / tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp8, tmp17, tmp18)
    tmp20 = tmp6 * tmp19
    tmp21 = tl.load(in_ptr5 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.exp(tmp21)
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 - tmp25
    tmp27 = tmp5 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp4, tmp27, tmp28)
    tmp30 = tmp0 >= tmp3
    tmp31 = tl.full([1], 2, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.load(in_ptr7 + (x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr8 + (x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp38.to(tl.int64)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp12 / tmp40
    tmp42 = tl.where(tmp36, tmp41, tmp18)
    tmp43 = tmp34 * tmp42
    tmp44 = tl.load(in_ptr11 + (x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.exp(tmp44)
    tmp48 = tmp45 * tmp47
    tmp49 = tmp43 - tmp48
    tmp50 = tmp33 + tmp49
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tl.where(tmp4, tmp29, tmp52)
    tl.store(out_ptr0 + (x2), tmp53, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lr/clrmorzbfi22u5tbhz2ldvohqagv3nfrqvsyqkbxykyuducb64jx.py
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
    size_hints=[8, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2*r2) + (256*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqwrcuiegvgzeqvw3oqwfwxngawzzojcfbf7dnakxg7asdugqxu.py
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
    size_hints=[2, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rt/crt6cma62frauxsloo734slp43q3ulc7k6qb372hmntdk6e7tmnc.py
# Source Nodes: [hidden_states_179], Original ATen: [aten.div, aten.mul, aten.sum]
# hidden_states_179 => div_48
triton_per_fused_div_mul_sum_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 / tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coelw3halbomdrdnd4klehcq3ijxgipq2zusbrzcszushmb53bcq.py
# Source Nodes: [hidden_states_179, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
# hidden_states_179 => div_48
# query_states => full_default_1
triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = -tmp2
    tmp6 = tmp4 / tmp5
    tmp7 = tmp6 / tmp5
    tmp8 = tmp3 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tmp2 / tmp5
    tmp14 = -tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = 2.0
    tmp20 = tmp5 * tmp19
    tmp21 = tmp12 / tmp20
    tmp22 = 768.0
    tmp23 = tmp21 / tmp22
    tmp24 = tmp4 * tmp19
    tmp25 = tmp23 * tmp24
    tmp26 = -tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp13 + tmp25
    tmp32 = tmp18 + tmp30
    tmp33 = tmp32 / tmp22
    tmp34 = tmp31 + tmp33
    tmp36 = 0.0
    tmp37 = tl.where(tmp35, tmp36, tmp34)
    tmp38 = 1.1111111111111112
    tmp39 = tmp37 * tmp38
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp34, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/62/c623a6irvbmaod7hhgpc6gexz2256fj7yqzsz4yanbcnevn7fkai.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vs/cvshiaozrzb6adzci7cftvfkiaxnsnwxzc3pfele7xt7cdvb2fls.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/mi/cmig47we7pe6qi4m6fodzi7o7v6bw622yrvfq45hd42yuxtarlpf.py
# Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_11 => add_107, erf_11, mul_108
triton_poi_fused_gelu_gelu_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/6y/c6yqq7sfjgaaxdkh5o3kqam47b6ywf4s5xn4mipbwryklhixmpic.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_10', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6ughq2lkyzherih2mehrcz7rlcve3vxx3kazka7vrbbtqebhve.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_11', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/p3/cp3prpymlxtxxztdi5tjuel27ec6f6puprrjjggcbgp7y5crcn3q.py
# Source Nodes: [hidden_states_171], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
# hidden_states_171 => div_47
triton_per_fused_add_div_mul_sum_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 / tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wv5coba22pfbszziwhoj2attgaywvjbldlcr7x4veuizubtswv.py
# Source Nodes: [hidden_states_171, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
# hidden_states_171 => div_47
# query_states => full_default_1
triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = -tmp4
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8 / tmp7
    tmp10 = tmp5 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tmp4 / tmp7
    tmp16 = -tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 2.0
    tmp22 = tmp7 * tmp21
    tmp23 = tmp14 / tmp22
    tmp24 = 768.0
    tmp25 = tmp23 / tmp24
    tmp26 = tmp6 * tmp21
    tmp27 = tmp25 * tmp26
    tmp28 = -tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp15 + tmp27
    tmp34 = tmp20 + tmp32
    tmp35 = tmp34 / tmp24
    tmp36 = tmp33 + tmp35
    tmp38 = 0.0
    tmp39 = tl.where(tmp37, tmp38, tmp36)
    tmp40 = 1.1111111111111112
    tmp41 = tmp39 * tmp40
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp36, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp41, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/64/c64zp2oscrf52aeincx3chlbr5lft67jb3mbzfsap2r6bz7u7gfa.py
# Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
# query_states => full_default_1
triton_per_fused__softmax_backward_data_masked_fill_mul_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_masked_fill_mul_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 6144
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jvvpl5lewn7jokptknxo4qix7sbjahhxaoekknpfyvw2d4oiav.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/di/cdik5a3dddfnv5u6jl4xqyob6uhk7gp7mlqjckysyjf5rj37tkv4.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_16', 'mutated_arg_names': []}
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
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (256*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwcpfvwxtemqmvw24wwgz6algkbmzcmbqxxhwrclaqhyujwnxmu.py
# Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
# scale => full_default_2
triton_red_fused_div_sqrt_sum_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_sqrt_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3apnmzw3jvxypqn3uk7vjnspvkw3eola76b2txt47milqfufg4l.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 192
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (64*y3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 8.0
    tmp7 = tmp5 / tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 128, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-32768) + y0 + (512*x2) + (32768*y1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tl.full([1, 1], 192, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr2 + ((-128) + x2 + (64*y3)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp13, tmp16, tmp22)
    tmp24 = tl.where(tmp4, tmp9, tmp23)
    tl.store(out_ptr0 + (x2 + (192*y1) + (2304*y0)), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/az/cazrqgeud7cqxq6kfu7mhntpqj4lv7mqvlkfmfkglqqky2755i2w.py
# Source Nodes: [hidden_states_1, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.sum]
# hidden_states_1 => div
# query_states => full_default_1
triton_per_fused_add_div_masked_fill_mul_sum_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_masked_fill_mul_sum_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = 1.1111111111111112
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp14 = tmp12 / tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cot6fs7z2bwwjyznctbwrq66b6hlz3rjrulm2t2qg6e5ulr7dbzk.py
# Source Nodes: [hidden_states_1, query_states], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
# hidden_states_1 => div
# query_states => full_default_1
triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*i64', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr5, out_ptr6, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp42 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = 1.1111111111111112
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 / tmp10
    tmp12 = -tmp9
    tmp14 = tmp13 / tmp10
    tmp15 = tmp14 / tmp10
    tmp16 = tmp12 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 2.0
    tmp22 = tmp10 * tmp21
    tmp23 = tmp20 / tmp22
    tmp24 = 768.0
    tmp25 = tmp23 / tmp24
    tmp26 = tmp13 * tmp21
    tmp27 = tmp25 * tmp26
    tmp28 = -tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = -tmp11
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = tmp11 + tmp27
    tmp39 = tmp37 + tmp32
    tmp40 = tmp39 / tmp24
    tmp41 = tmp38 + tmp40
    tmp43 = tl.full([1], -1, tl.int64)
    tmp44 = tmp42 == tmp43
    tmp45 = tl.where(tmp44, tmp4, tmp41)
    tmp47 = tl.full([1], 0, tl.int64)
    tmp48 = tmp46 == tmp47
    tmp49 = tl.where(tmp48, tmp4, tmp41)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp45, rmask & xmask)
    tl.store(out_ptr6 + (r1 + (768*x0)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czszd4ko2keesyc2gyntcrttl2dyechv6p6vgblq6mzeneujccq6.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgxnlwi4o3fot76ceku3alfyvcpjophp5tcynlyhjw5hxzsbsery.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38603520
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
    primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_164, slice_1, sub, sqrt, convert_element_type, view, convert_element_type_2, view_12, convert_element_type_3, sub_6, sqrt_2, view_14, addmm_1, view_16, convert_element_type_4, sub_9, sqrt_3, view_18, convert_element_type_6, view_30, convert_element_type_7, sub_14, sqrt_5, view_32, addmm_4, view_34, convert_element_type_8, sub_17, sqrt_6, view_36, convert_element_type_10, view_48, convert_element_type_11, sub_22, sqrt_8, view_50, addmm_7, view_52, convert_element_type_12, sub_25, sqrt_9, view_54, convert_element_type_14, view_66, convert_element_type_15, sub_30, sqrt_11, view_68, addmm_10, view_70, convert_element_type_16, sub_33, sqrt_12, view_72, convert_element_type_18, view_84, convert_element_type_19, sub_38, sqrt_14, view_86, addmm_13, view_88, convert_element_type_20, sub_41, sqrt_15, view_90, convert_element_type_22, view_102, convert_element_type_23, sub_46, sqrt_17, view_104, addmm_16, view_106, convert_element_type_24, sub_49, sqrt_18, view_108, convert_element_type_26, view_120, convert_element_type_27, sub_54, sqrt_20, view_122, addmm_19, view_124, convert_element_type_28, sub_57, sqrt_21, view_126, convert_element_type_30, view_138, convert_element_type_31, sub_62, sqrt_23, view_140, addmm_22, view_142, convert_element_type_32, sub_65, sqrt_24, view_144, convert_element_type_34, view_156, convert_element_type_35, sub_70, sqrt_26, view_158, addmm_25, view_160, convert_element_type_36, sub_73, sqrt_27, view_162, convert_element_type_38, view_174, convert_element_type_39, sub_78, sqrt_29, view_176, addmm_28, view_178, convert_element_type_40, sub_81, sqrt_30, view_180, convert_element_type_42, view_192, convert_element_type_43, sub_86, sqrt_32, view_194, addmm_31, view_196, convert_element_type_44, sub_89, sqrt_33, view_198, convert_element_type_46, view_210, convert_element_type_47, sub_94, sqrt_35, view_212, addmm_34, view_214, convert_element_type_48, sub_97, sqrt_36, view_216, sub_100, ne, sub_102, ne_3, ne_6, where_65, ne_8, where_67, permute_146, permute_150, permute_154, permute_158, permute_163, permute_164, alias_45, permute_165, permute_166, permute_173, permute_175, permute_179, permute_183, permute_188, permute_189, alias_50, permute_190, permute_191, permute_198, permute_200, permute_204, permute_208, permute_213, permute_214, alias_55, permute_215, permute_216, permute_223, permute_225, permute_229, permute_233, permute_238, permute_239, alias_60, permute_240, permute_241, permute_248, permute_250, permute_254, permute_258, permute_263, permute_264, alias_65, permute_265, permute_266, permute_273, permute_275, permute_279, permute_283, permute_288, permute_289, alias_70, permute_290, permute_291, permute_298, permute_300, permute_304, permute_308, permute_313, permute_314, alias_75, permute_315, permute_316, permute_323, permute_325, permute_329, permute_333, permute_338, permute_339, alias_80, permute_340, permute_341, permute_348, permute_350, permute_354, permute_358, permute_363, permute_364, alias_85, permute_365, permute_366, permute_373, permute_375, permute_379, permute_383, permute_388, permute_389, alias_90, permute_390, permute_391, permute_398, permute_400, permute_404, permute_408, permute_413, permute_414, alias_95, permute_415, permute_416, permute_423, permute_425, permute_429, permute_433, permute_438, permute_439, alias_100, permute_440, permute_441, permute_448, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_164, (1, 512), (512, 1))
    assert_size_stride(slice_1, (1, 512), (512, 1))
    assert_size_stride(sub, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt, (1, 512, 1), (512, 1, 1))
    assert_size_stride(convert_element_type, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_2, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_12, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_6, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_2, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_14, (512, 768), (768, 1))
    assert_size_stride(addmm_1, (512, 3072), (3072, 1))
    assert_size_stride(view_16, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_4, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_9, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_3, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_6, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_30, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_14, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_5, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_32, (512, 768), (768, 1))
    assert_size_stride(addmm_4, (512, 3072), (3072, 1))
    assert_size_stride(view_34, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_6, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_36, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_10, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_48, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_22, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_8, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_50, (512, 768), (768, 1))
    assert_size_stride(addmm_7, (512, 3072), (3072, 1))
    assert_size_stride(view_52, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_12, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_25, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_9, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_54, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_14, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_15, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_30, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_11, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_68, (512, 768), (768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(view_70, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_16, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_33, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_12, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_72, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_18, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_84, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_19, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_38, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_14, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_86, (512, 768), (768, 1))
    assert_size_stride(addmm_13, (512, 3072), (3072, 1))
    assert_size_stride(view_88, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_20, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_15, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_90, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_22, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_102, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_23, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_46, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_17, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(addmm_16, (512, 3072), (3072, 1))
    assert_size_stride(view_106, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_49, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_18, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_108, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_26, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_120, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_54, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_20, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_122, (512, 768), (768, 1))
    assert_size_stride(addmm_19, (512, 3072), (3072, 1))
    assert_size_stride(view_124, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_28, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_21, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_126, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_30, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_138, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_62, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_23, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_140, (512, 768), (768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(view_142, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_32, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_65, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_24, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_144, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_34, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_156, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_35, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_70, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_158, (512, 768), (768, 1))
    assert_size_stride(addmm_25, (512, 3072), (3072, 1))
    assert_size_stride(view_160, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_162, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_38, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_174, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_39, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_78, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_29, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 3072), (3072, 1))
    assert_size_stride(view_178, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_40, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_180, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_42, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_192, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_43, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_86, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_32, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_194, (512, 768), (768, 1))
    assert_size_stride(addmm_31, (512, 3072), (3072, 1))
    assert_size_stride(view_196, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_44, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_89, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_198, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_46, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_210, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_47, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_94, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_35, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_212, (512, 768), (768, 1))
    assert_size_stride(addmm_34, (512, 3072), (3072, 1))
    assert_size_stride(view_214, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_48, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(sub_100, (1, 512), (512, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_102, (1, 512), (512, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_65, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_67, (1, 1), (1, 1))
    assert_size_stride(permute_146, (2, 768), (768, 1))
    assert_size_stride(permute_150, (768, 3072), (3072, 1))
    assert_size_stride(permute_154, (3072, 768), (768, 1))
    assert_size_stride(permute_158, (768, 768), (768, 1))
    assert_size_stride(permute_163, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_164, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_45, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_165, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_166, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_173, (2304, 768), (768, 1))
    assert_size_stride(permute_175, (768, 3072), (3072, 1))
    assert_size_stride(permute_179, (3072, 768), (768, 1))
    assert_size_stride(permute_183, (768, 768), (768, 1))
    assert_size_stride(permute_188, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_189, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_50, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_190, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_191, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_198, (2304, 768), (768, 1))
    assert_size_stride(permute_200, (768, 3072), (3072, 1))
    assert_size_stride(permute_204, (3072, 768), (768, 1))
    assert_size_stride(permute_208, (768, 768), (768, 1))
    assert_size_stride(permute_213, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_214, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_55, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_215, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_216, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_223, (2304, 768), (768, 1))
    assert_size_stride(permute_225, (768, 3072), (3072, 1))
    assert_size_stride(permute_229, (3072, 768), (768, 1))
    assert_size_stride(permute_233, (768, 768), (768, 1))
    assert_size_stride(permute_238, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_239, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_60, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_240, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_241, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_248, (2304, 768), (768, 1))
    assert_size_stride(permute_250, (768, 3072), (3072, 1))
    assert_size_stride(permute_254, (3072, 768), (768, 1))
    assert_size_stride(permute_258, (768, 768), (768, 1))
    assert_size_stride(permute_263, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_264, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_65, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_265, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_266, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_273, (2304, 768), (768, 1))
    assert_size_stride(permute_275, (768, 3072), (3072, 1))
    assert_size_stride(permute_279, (3072, 768), (768, 1))
    assert_size_stride(permute_283, (768, 768), (768, 1))
    assert_size_stride(permute_288, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_289, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_70, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_290, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_291, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_298, (2304, 768), (768, 1))
    assert_size_stride(permute_300, (768, 3072), (3072, 1))
    assert_size_stride(permute_304, (3072, 768), (768, 1))
    assert_size_stride(permute_308, (768, 768), (768, 1))
    assert_size_stride(permute_313, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_314, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_75, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_315, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_316, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_323, (2304, 768), (768, 1))
    assert_size_stride(permute_325, (768, 3072), (3072, 1))
    assert_size_stride(permute_329, (3072, 768), (768, 1))
    assert_size_stride(permute_333, (768, 768), (768, 1))
    assert_size_stride(permute_338, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_339, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_80, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_340, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_341, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_348, (2304, 768), (768, 1))
    assert_size_stride(permute_350, (768, 3072), (3072, 1))
    assert_size_stride(permute_354, (3072, 768), (768, 1))
    assert_size_stride(permute_358, (768, 768), (768, 1))
    assert_size_stride(permute_363, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_364, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_85, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_365, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_366, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_373, (2304, 768), (768, 1))
    assert_size_stride(permute_375, (768, 3072), (3072, 1))
    assert_size_stride(permute_379, (3072, 768), (768, 1))
    assert_size_stride(permute_383, (768, 768), (768, 1))
    assert_size_stride(permute_388, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_389, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_90, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_390, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_391, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_398, (2304, 768), (768, 1))
    assert_size_stride(permute_400, (768, 3072), (3072, 1))
    assert_size_stride(permute_404, (3072, 768), (768, 1))
    assert_size_stride(permute_408, (768, 768), (768, 1))
    assert_size_stride(permute_413, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_414, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_95, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_415, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_416, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_423, (2304, 768), (768, 1))
    assert_size_stride(permute_425, (768, 3072), (3072, 1))
    assert_size_stride(permute_429, (3072, 768), (768, 1))
    assert_size_stride(permute_433, (768, 768), (768, 1))
    assert_size_stride(permute_438, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_439, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_100, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_440, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_441, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_448, (2304, 768), (768, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512), (512, 1))
    assert_size_stride(tangents_3, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_0.run(buf0, 512, grid=grid(512), stream=stream0)
        aten.scatter_(buf0,1,where_65,-1.0)
        del where_65
        buf4 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_0.run(buf4, 512, grid=grid(512), stream=stream0)
        aten.scatter_(buf4,1,where_67,-1.0)
        del where_67
        buf3 = empty((1, 1), device='cuda', dtype=torch.float32)
        buf7 = empty((1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_loss, query_states, start_loss], Original ATen: [aten._log_softmax_backward_data, aten.div, aten.masked_fill, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_per_fused__log_softmax_backward_data_div_masked_fill_nll_loss_backward_nll_loss_forward_1.run(buf0, ne_6, tangents_1, ne_3, buf4, ne_8, ne, buf3, buf7, 1, 512, grid=grid(1), stream=stream0)
        buf8 = empty((1, 512, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_2.run(tangents_2, buf4, ne_8, tangents_1, ne, sub_100, buf7, tangents_3, buf0, ne_6, ne_3, sub_102, buf3, buf8, 1024, grid=grid(1024), stream=stream0)
        del buf0
        del buf3
        del buf4
        del buf7
        del ne
        del ne_3
        del ne_6
        del ne_8
        del sub_100
        del sub_102
        del tangents_1
        del tangents_2
        del tangents_3
        buf9 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (512, 2), (2, 1), 0), permute_146, out=buf9)
        del permute_146
        buf10 = empty((2, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (2, 512), (1, 2), 0), view_216, out=buf10)
        del view_216
        buf11 = empty_strided((1, 2, 4), (8, 1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf8, buf11, 8, 128, grid=grid(8), stream=stream0)
        del buf8
        buf12 = empty((1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf11, buf12, 2, 4, grid=grid(2), stream=stream0)
        del buf11
        buf13 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf14 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_179], Original ATen: [aten.div, aten.mul, aten.sum]
        triton_per_fused_div_mul_sum_5.run(buf9, sub_97, sqrt_36, buf13, buf14, 768, 512, grid=grid(768), stream=stream0)
        buf18 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf19 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_179, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_6.run(buf9, primals_73, sub_97, sqrt_36, convert_element_type_48, buf18, buf19, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_48
        del primals_73
        del sqrt_36
        del sub_97
        buf20 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (512, 768), (768, 1), 0), permute_150, out=buf20)
        del permute_150
        buf21 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (768, 512), (1, 768), 0), view_214, out=buf21)
        del view_214
        buf22 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf19, buf22, 3072, 128, grid=grid(3072), stream=stream0)
        buf23 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf22, buf23, 768, 4, grid=grid(768), stream=stream0)
        buf24 = reinterpret_tensor(buf20, (1, 512, 3072), (1572864, 3072, 1), 0); del buf20  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf24, addmm_34, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_34
        buf25 = reinterpret_tensor(buf19, (512, 768), (768, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (512, 3072), (3072, 1), 0), permute_154, out=buf25)
        del permute_154
        buf26 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (3072, 512), (1, 3072), 0), view_212, out=buf26)
        del view_212
        buf27 = empty_strided((1, 3072, 4), (12288, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf24, buf27, 12288, 128, grid=grid(12288), stream=stream0)
        buf28 = reinterpret_tensor(buf22, (1, 3072), (3072, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf27, buf28, 3072, 4, grid=grid(3072), stream=stream0)
        buf29 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf30 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_171], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf18, buf25, sub_94, sqrt_35, buf29, buf30, 768, 512, grid=grid(768), stream=stream0)
        buf34 = reinterpret_tensor(buf9, (1, 512, 768), (393216, 768, 1), 0); del buf9  # reuse
        buf35 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_171, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf18, buf25, primals_71, sub_94, sqrt_35, convert_element_type_47, buf34, buf35, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_47
        del primals_71
        del sqrt_35
        del sub_94
        buf36 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (512, 768), (768, 1), 0), permute_158, out=buf36)
        del permute_158
        buf37 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (768, 512), (1, 768), 0), view_210, out=buf37)
        del view_210
        buf38 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf35, buf38, 3072, 128, grid=grid(3072), stream=stream0)
        buf39 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf38, buf39, 768, 4, grid=grid(768), stream=stream0)
        buf40 = reinterpret_tensor(buf35, (12, 512, 64), (32768, 64, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_163, reinterpret_tensor(buf36, (12, 512, 64), (64, 768, 1), 0), out=buf40)
        del permute_163
        buf41 = empty((12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (12, 512, 64), (64, 768, 1), 0), permute_164, out=buf41)
        del permute_164
        buf43 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_46, buf41, alias_45, buf43, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_45
        del convert_element_type_46
        buf44 = reinterpret_tensor(buf36, (12, 64, 512), (32768, 512, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_165, reinterpret_tensor(buf43, (12, 512, 512), (262144, 512, 1), 0), out=buf44)
        del permute_165
        buf45 = reinterpret_tensor(buf18, (12, 512, 64), (32768, 64, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (12, 512, 512), (262144, 512, 1), 0), permute_166, out=buf45)
        del permute_166
        buf46 = reinterpret_tensor(buf38, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf40, buf46, 3072, 128, grid=grid(3072), stream=stream0)
        buf47 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf46, buf47, 768, 4, grid=grid(768), stream=stream0)
        buf48 = buf46; del buf46  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf45, buf48, 3072, 128, grid=grid(3072), stream=stream0)
        buf49 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf48, buf49, 768, 4, grid=grid(768), stream=stream0)
        buf50 = empty((1, 512, 12, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf45, buf44, buf40, buf50, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf51 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (2304, 512), (1, 2304), 0), view_198, out=buf51)
        del view_198
        buf52 = reinterpret_tensor(buf45, (512, 768), (768, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (512, 2304), (2304, 1), 0), permute_173, out=buf52)
        del permute_173
        buf53 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf54 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_164], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf34, buf52, sub_89, sqrt_33, buf53, buf54, 768, 512, grid=grid(768), stream=stream0)
        buf58 = reinterpret_tensor(buf44, (1, 512, 768), (393216, 768, 1), 0); del buf44  # reuse
        buf59 = reinterpret_tensor(buf40, (1, 512, 768), (393216, 768, 1), 0); del buf40  # reuse
        # Source Nodes: [hidden_states_164, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf34, buf52, primals_67, sub_89, sqrt_33, convert_element_type_44, buf58, buf59, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_44
        del primals_67
        del sqrt_33
        del sub_89
        buf60 = reinterpret_tensor(buf24, (512, 3072), (3072, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (512, 768), (768, 1), 0), permute_175, out=buf60)
        del permute_175
        buf61 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (768, 512), (1, 768), 0), view_196, out=buf61)
        del view_196
        buf62 = reinterpret_tensor(buf48, (1, 768, 4), (3072, 1, 768), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf59, buf62, 3072, 128, grid=grid(3072), stream=stream0)
        buf63 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf62, buf63, 768, 4, grid=grid(768), stream=stream0)
        buf64 = reinterpret_tensor(buf60, (1, 512, 3072), (1572864, 3072, 1), 0); del buf60  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf64, addmm_31, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_31
        buf65 = reinterpret_tensor(buf59, (512, 768), (768, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 3072), (3072, 1), 0), permute_179, out=buf65)
        del permute_179
        buf66 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (3072, 512), (1, 3072), 0), view_194, out=buf66)
        del view_194
        buf67 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf64, buf67, 12288, 128, grid=grid(12288), stream=stream0)
        buf68 = reinterpret_tensor(buf62, (1, 3072), (3072, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf67, buf68, 3072, 4, grid=grid(3072), stream=stream0)
        buf69 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf70 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_156], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf58, buf65, sub_86, sqrt_32, buf69, buf70, 768, 512, grid=grid(768), stream=stream0)
        buf74 = reinterpret_tensor(buf52, (1, 512, 768), (393216, 768, 1), 0); del buf52  # reuse
        buf75 = buf34; del buf34  # reuse
        # Source Nodes: [hidden_states_156, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf58, buf65, primals_65, sub_86, sqrt_32, convert_element_type_43, buf74, buf75, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_43
        del primals_65
        del sqrt_32
        del sub_86
        buf76 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (512, 768), (768, 1), 0), permute_183, out=buf76)
        del permute_183
        buf77 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (768, 512), (1, 768), 0), view_192, out=buf77)
        del view_192
        buf78 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf75, buf78, 3072, 128, grid=grid(3072), stream=stream0)
        buf79 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf78, buf79, 768, 4, grid=grid(768), stream=stream0)
        buf80 = reinterpret_tensor(buf75, (12, 512, 64), (32768, 64, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_188, reinterpret_tensor(buf76, (12, 512, 64), (64, 768, 1), 0), out=buf80)
        del permute_188
        buf81 = reinterpret_tensor(buf43, (12, 512, 512), (262144, 512, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf76, (12, 512, 64), (64, 768, 1), 0), permute_189, out=buf81)
        del permute_189
        buf83 = reinterpret_tensor(buf41, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf41  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_42, buf81, alias_50, buf83, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_50
        del convert_element_type_42
        buf84 = reinterpret_tensor(buf76, (12, 64, 512), (32768, 512, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_190, reinterpret_tensor(buf83, (12, 512, 512), (262144, 512, 1), 0), out=buf84)
        del permute_190
        buf85 = reinterpret_tensor(buf58, (12, 512, 64), (32768, 64, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf83, (12, 512, 512), (262144, 512, 1), 0), permute_191, out=buf85)
        del permute_191
        buf86 = reinterpret_tensor(buf78, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf80, buf86, 3072, 128, grid=grid(3072), stream=stream0)
        buf87 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf86, buf87, 768, 4, grid=grid(768), stream=stream0)
        buf88 = buf86; del buf86  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf85, buf88, 3072, 128, grid=grid(3072), stream=stream0)
        buf89 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf88, buf89, 768, 4, grid=grid(768), stream=stream0)
        buf90 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf85, buf84, buf80, buf90, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf91 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (2304, 512), (1, 2304), 0), view_180, out=buf91)
        del view_180
        buf92 = reinterpret_tensor(buf85, (512, 768), (768, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (512, 2304), (2304, 1), 0), permute_198, out=buf92)
        del permute_198
        buf93 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf94 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_149], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf74, buf92, sub_81, sqrt_30, buf93, buf94, 768, 512, grid=grid(768), stream=stream0)
        buf98 = reinterpret_tensor(buf84, (1, 512, 768), (393216, 768, 1), 0); del buf84  # reuse
        buf99 = reinterpret_tensor(buf80, (1, 512, 768), (393216, 768, 1), 0); del buf80  # reuse
        # Source Nodes: [hidden_states_149, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf74, buf92, primals_61, sub_81, sqrt_30, convert_element_type_40, buf98, buf99, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_40
        del primals_61
        del sqrt_30
        del sub_81
        buf100 = reinterpret_tensor(buf64, (512, 3072), (3072, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 768), (768, 1), 0), permute_200, out=buf100)
        del permute_200
        buf101 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (768, 512), (1, 768), 0), view_178, out=buf101)
        del view_178
        buf102 = reinterpret_tensor(buf88, (1, 768, 4), (3072, 1, 768), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf99, buf102, 3072, 128, grid=grid(3072), stream=stream0)
        buf103 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf102, buf103, 768, 4, grid=grid(768), stream=stream0)
        buf104 = reinterpret_tensor(buf100, (1, 512, 3072), (1572864, 3072, 1), 0); del buf100  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf104, addmm_28, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_28
        buf105 = reinterpret_tensor(buf99, (512, 768), (768, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (512, 3072), (3072, 1), 0), permute_204, out=buf105)
        del permute_204
        buf106 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (3072, 512), (1, 3072), 0), view_176, out=buf106)
        del view_176
        buf107 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf104, buf107, 12288, 128, grid=grid(12288), stream=stream0)
        buf108 = reinterpret_tensor(buf102, (1, 3072), (3072, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf107, buf108, 3072, 4, grid=grid(3072), stream=stream0)
        buf109 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf110 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_141], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf98, buf105, sub_78, sqrt_29, buf109, buf110, 768, 512, grid=grid(768), stream=stream0)
        buf114 = reinterpret_tensor(buf92, (1, 512, 768), (393216, 768, 1), 0); del buf92  # reuse
        buf115 = buf74; del buf74  # reuse
        # Source Nodes: [hidden_states_141, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf98, buf105, primals_59, sub_78, sqrt_29, convert_element_type_39, buf114, buf115, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_39
        del primals_59
        del sqrt_29
        del sub_78
        buf116 = reinterpret_tensor(buf98, (512, 768), (768, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (512, 768), (768, 1), 0), permute_208, out=buf116)
        del permute_208
        buf117 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (768, 512), (1, 768), 0), view_174, out=buf117)
        del view_174
        buf118 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf115, buf118, 3072, 128, grid=grid(3072), stream=stream0)
        buf119 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf118, buf119, 768, 4, grid=grid(768), stream=stream0)
        buf120 = reinterpret_tensor(buf115, (12, 512, 64), (32768, 64, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_213, reinterpret_tensor(buf116, (12, 512, 64), (64, 768, 1), 0), out=buf120)
        del permute_213
        buf121 = reinterpret_tensor(buf83, (12, 512, 512), (262144, 512, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (12, 512, 64), (64, 768, 1), 0), permute_214, out=buf121)
        del permute_214
        buf123 = reinterpret_tensor(buf81, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf81  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_38, buf121, alias_55, buf123, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_55
        del convert_element_type_38
        buf124 = reinterpret_tensor(buf116, (12, 64, 512), (32768, 512, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_215, reinterpret_tensor(buf123, (12, 512, 512), (262144, 512, 1), 0), out=buf124)
        del permute_215
        buf125 = reinterpret_tensor(buf105, (12, 512, 64), (32768, 64, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf123, (12, 512, 512), (262144, 512, 1), 0), permute_216, out=buf125)
        del permute_216
        buf126 = reinterpret_tensor(buf118, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf120, buf126, 3072, 128, grid=grid(3072), stream=stream0)
        buf127 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf126, buf127, 768, 4, grid=grid(768), stream=stream0)
        buf128 = buf126; del buf126  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf125, buf128, 3072, 128, grid=grid(3072), stream=stream0)
        buf129 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf128, buf129, 768, 4, grid=grid(768), stream=stream0)
        buf130 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf125, buf124, buf120, buf130, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf131 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (2304, 512), (1, 2304), 0), view_162, out=buf131)
        del view_162
        buf132 = reinterpret_tensor(buf125, (512, 768), (768, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (512, 2304), (2304, 1), 0), permute_223, out=buf132)
        del permute_223
        buf133 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf134 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf114, buf132, sub_73, sqrt_27, buf133, buf134, 768, 512, grid=grid(768), stream=stream0)
        buf138 = reinterpret_tensor(buf124, (1, 512, 768), (393216, 768, 1), 0); del buf124  # reuse
        buf139 = reinterpret_tensor(buf120, (1, 512, 768), (393216, 768, 1), 0); del buf120  # reuse
        # Source Nodes: [hidden_states_134, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf114, buf132, primals_55, sub_73, sqrt_27, convert_element_type_36, buf138, buf139, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_36
        del primals_55
        del sqrt_27
        del sub_73
        buf140 = reinterpret_tensor(buf104, (512, 3072), (3072, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (512, 768), (768, 1), 0), permute_225, out=buf140)
        del permute_225
        buf141 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (768, 512), (1, 768), 0), view_160, out=buf141)
        del view_160
        buf142 = reinterpret_tensor(buf128, (1, 768, 4), (3072, 1, 768), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf139, buf142, 3072, 128, grid=grid(3072), stream=stream0)
        buf143 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf142, buf143, 768, 4, grid=grid(768), stream=stream0)
        buf144 = reinterpret_tensor(buf140, (1, 512, 3072), (1572864, 3072, 1), 0); del buf140  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf144, addmm_25, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_25
        buf145 = reinterpret_tensor(buf139, (512, 768), (768, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (512, 3072), (3072, 1), 0), permute_229, out=buf145)
        del permute_229
        buf146 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (3072, 512), (1, 3072), 0), view_158, out=buf146)
        del view_158
        buf147 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf144, buf147, 12288, 128, grid=grid(12288), stream=stream0)
        buf148 = reinterpret_tensor(buf142, (1, 3072), (3072, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf147, buf148, 3072, 4, grid=grid(3072), stream=stream0)
        buf149 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf150 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_126], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf138, buf145, sub_70, sqrt_26, buf149, buf150, 768, 512, grid=grid(768), stream=stream0)
        buf154 = reinterpret_tensor(buf132, (1, 512, 768), (393216, 768, 1), 0); del buf132  # reuse
        buf155 = buf114; del buf114  # reuse
        # Source Nodes: [hidden_states_126, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf138, buf145, primals_53, sub_70, sqrt_26, convert_element_type_35, buf154, buf155, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_35
        del primals_53
        del sqrt_26
        del sub_70
        buf156 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (512, 768), (768, 1), 0), permute_233, out=buf156)
        del permute_233
        buf157 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (768, 512), (1, 768), 0), view_156, out=buf157)
        del view_156
        buf158 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf155, buf158, 3072, 128, grid=grid(3072), stream=stream0)
        buf159 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf158, buf159, 768, 4, grid=grid(768), stream=stream0)
        buf160 = reinterpret_tensor(buf155, (12, 512, 64), (32768, 64, 1), 0); del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_238, reinterpret_tensor(buf156, (12, 512, 64), (64, 768, 1), 0), out=buf160)
        del permute_238
        buf161 = reinterpret_tensor(buf123, (12, 512, 512), (262144, 512, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf156, (12, 512, 64), (64, 768, 1), 0), permute_239, out=buf161)
        del permute_239
        buf163 = reinterpret_tensor(buf121, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf121  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_34, buf161, alias_60, buf163, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_60
        del convert_element_type_34
        buf164 = reinterpret_tensor(buf156, (12, 64, 512), (32768, 512, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_240, reinterpret_tensor(buf163, (12, 512, 512), (262144, 512, 1), 0), out=buf164)
        del permute_240
        buf165 = reinterpret_tensor(buf138, (12, 512, 64), (32768, 64, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf163, (12, 512, 512), (262144, 512, 1), 0), permute_241, out=buf165)
        del permute_241
        buf166 = reinterpret_tensor(buf158, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf160, buf166, 3072, 128, grid=grid(3072), stream=stream0)
        buf167 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf166, buf167, 768, 4, grid=grid(768), stream=stream0)
        buf168 = buf166; del buf166  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf165, buf168, 3072, 128, grid=grid(3072), stream=stream0)
        buf169 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf168, buf169, 768, 4, grid=grid(768), stream=stream0)
        buf170 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf165, buf164, buf160, buf170, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf171 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (2304, 512), (1, 2304), 0), view_144, out=buf171)
        del view_144
        buf172 = reinterpret_tensor(buf165, (512, 768), (768, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (512, 2304), (2304, 1), 0), permute_248, out=buf172)
        del permute_248
        buf173 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf174 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_119], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf154, buf172, sub_65, sqrt_24, buf173, buf174, 768, 512, grid=grid(768), stream=stream0)
        buf178 = reinterpret_tensor(buf164, (1, 512, 768), (393216, 768, 1), 0); del buf164  # reuse
        buf179 = reinterpret_tensor(buf160, (1, 512, 768), (393216, 768, 1), 0); del buf160  # reuse
        # Source Nodes: [hidden_states_119, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf154, buf172, primals_49, sub_65, sqrt_24, convert_element_type_32, buf178, buf179, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_32
        del primals_49
        del sqrt_24
        del sub_65
        buf180 = reinterpret_tensor(buf144, (512, 3072), (3072, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (512, 768), (768, 1), 0), permute_250, out=buf180)
        del permute_250
        buf181 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (768, 512), (1, 768), 0), view_142, out=buf181)
        del view_142
        buf182 = reinterpret_tensor(buf168, (1, 768, 4), (3072, 1, 768), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf179, buf182, 3072, 128, grid=grid(3072), stream=stream0)
        buf183 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf182, buf183, 768, 4, grid=grid(768), stream=stream0)
        buf184 = reinterpret_tensor(buf180, (1, 512, 3072), (1572864, 3072, 1), 0); del buf180  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf184, addmm_22, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_22
        buf185 = reinterpret_tensor(buf179, (512, 768), (768, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf184, (512, 3072), (3072, 1), 0), permute_254, out=buf185)
        del permute_254
        buf186 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf184, (3072, 512), (1, 3072), 0), view_140, out=buf186)
        del view_140
        buf187 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf184, buf187, 12288, 128, grid=grid(12288), stream=stream0)
        buf188 = reinterpret_tensor(buf182, (1, 3072), (3072, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf187, buf188, 3072, 4, grid=grid(3072), stream=stream0)
        buf189 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf190 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_111], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf178, buf185, sub_62, sqrt_23, buf189, buf190, 768, 512, grid=grid(768), stream=stream0)
        buf194 = reinterpret_tensor(buf172, (1, 512, 768), (393216, 768, 1), 0); del buf172  # reuse
        buf195 = buf154; del buf154  # reuse
        # Source Nodes: [hidden_states_111, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf178, buf185, primals_47, sub_62, sqrt_23, convert_element_type_31, buf194, buf195, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_31
        del primals_47
        del sqrt_23
        del sub_62
        buf196 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 768), (768, 1), 0), permute_258, out=buf196)
        del permute_258
        buf197 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (768, 512), (1, 768), 0), view_138, out=buf197)
        del view_138
        buf198 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf195, buf198, 3072, 128, grid=grid(3072), stream=stream0)
        buf199 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf198, buf199, 768, 4, grid=grid(768), stream=stream0)
        buf200 = reinterpret_tensor(buf195, (12, 512, 64), (32768, 64, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_263, reinterpret_tensor(buf196, (12, 512, 64), (64, 768, 1), 0), out=buf200)
        del permute_263
        buf201 = reinterpret_tensor(buf163, (12, 512, 512), (262144, 512, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf196, (12, 512, 64), (64, 768, 1), 0), permute_264, out=buf201)
        del permute_264
        buf203 = reinterpret_tensor(buf161, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf161  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_30, buf201, alias_65, buf203, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_65
        del convert_element_type_30
        buf204 = reinterpret_tensor(buf196, (12, 64, 512), (32768, 512, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_265, reinterpret_tensor(buf203, (12, 512, 512), (262144, 512, 1), 0), out=buf204)
        del permute_265
        buf205 = reinterpret_tensor(buf178, (12, 512, 64), (32768, 64, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf203, (12, 512, 512), (262144, 512, 1), 0), permute_266, out=buf205)
        del permute_266
        buf206 = reinterpret_tensor(buf198, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf200, buf206, 3072, 128, grid=grid(3072), stream=stream0)
        buf207 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf206, buf207, 768, 4, grid=grid(768), stream=stream0)
        buf208 = buf206; del buf206  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf205, buf208, 3072, 128, grid=grid(3072), stream=stream0)
        buf209 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf208, buf209, 768, 4, grid=grid(768), stream=stream0)
        buf210 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf205, buf204, buf200, buf210, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf211 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (2304, 512), (1, 2304), 0), view_126, out=buf211)
        del view_126
        buf212 = reinterpret_tensor(buf205, (512, 768), (768, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (512, 2304), (2304, 1), 0), permute_273, out=buf212)
        del permute_273
        buf213 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf214 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_104], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf194, buf212, sub_57, sqrt_21, buf213, buf214, 768, 512, grid=grid(768), stream=stream0)
        buf218 = reinterpret_tensor(buf204, (1, 512, 768), (393216, 768, 1), 0); del buf204  # reuse
        buf219 = reinterpret_tensor(buf200, (1, 512, 768), (393216, 768, 1), 0); del buf200  # reuse
        # Source Nodes: [hidden_states_104, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf194, buf212, primals_43, sub_57, sqrt_21, convert_element_type_28, buf218, buf219, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_28
        del primals_43
        del sqrt_21
        del sub_57
        buf220 = reinterpret_tensor(buf184, (512, 3072), (3072, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (512, 768), (768, 1), 0), permute_275, out=buf220)
        del permute_275
        buf221 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (768, 512), (1, 768), 0), view_124, out=buf221)
        del view_124
        buf222 = reinterpret_tensor(buf208, (1, 768, 4), (3072, 1, 768), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf219, buf222, 3072, 128, grid=grid(3072), stream=stream0)
        buf223 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf222, buf223, 768, 4, grid=grid(768), stream=stream0)
        buf224 = reinterpret_tensor(buf220, (1, 512, 3072), (1572864, 3072, 1), 0); del buf220  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf224, addmm_19, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_19
        buf225 = reinterpret_tensor(buf219, (512, 768), (768, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (512, 3072), (3072, 1), 0), permute_279, out=buf225)
        del permute_279
        buf226 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (3072, 512), (1, 3072), 0), view_122, out=buf226)
        del view_122
        buf227 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf224, buf227, 12288, 128, grid=grid(12288), stream=stream0)
        buf228 = reinterpret_tensor(buf222, (1, 3072), (3072, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf227, buf228, 3072, 4, grid=grid(3072), stream=stream0)
        buf229 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf230 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_96], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf218, buf225, sub_54, sqrt_20, buf229, buf230, 768, 512, grid=grid(768), stream=stream0)
        buf234 = reinterpret_tensor(buf212, (1, 512, 768), (393216, 768, 1), 0); del buf212  # reuse
        buf235 = buf194; del buf194  # reuse
        # Source Nodes: [hidden_states_96, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf218, buf225, primals_41, sub_54, sqrt_20, convert_element_type_27, buf234, buf235, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_27
        del primals_41
        del sqrt_20
        del sub_54
        buf236 = buf225; del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (512, 768), (768, 1), 0), permute_283, out=buf236)
        del permute_283
        buf237 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (768, 512), (1, 768), 0), view_120, out=buf237)
        del view_120
        buf238 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf235, buf238, 3072, 128, grid=grid(3072), stream=stream0)
        buf239 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf238, buf239, 768, 4, grid=grid(768), stream=stream0)
        buf240 = reinterpret_tensor(buf235, (12, 512, 64), (32768, 64, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_288, reinterpret_tensor(buf236, (12, 512, 64), (64, 768, 1), 0), out=buf240)
        del permute_288
        buf241 = reinterpret_tensor(buf203, (12, 512, 512), (262144, 512, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf236, (12, 512, 64), (64, 768, 1), 0), permute_289, out=buf241)
        del permute_289
        buf243 = reinterpret_tensor(buf201, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf201  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_26, buf241, alias_70, buf243, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_70
        del convert_element_type_26
        buf244 = reinterpret_tensor(buf236, (12, 64, 512), (32768, 512, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_290, reinterpret_tensor(buf243, (12, 512, 512), (262144, 512, 1), 0), out=buf244)
        del permute_290
        buf245 = reinterpret_tensor(buf218, (12, 512, 64), (32768, 64, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf243, (12, 512, 512), (262144, 512, 1), 0), permute_291, out=buf245)
        del permute_291
        buf246 = reinterpret_tensor(buf238, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf240, buf246, 3072, 128, grid=grid(3072), stream=stream0)
        buf247 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf246, buf247, 768, 4, grid=grid(768), stream=stream0)
        buf248 = buf246; del buf246  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf245, buf248, 3072, 128, grid=grid(3072), stream=stream0)
        buf249 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf248, buf249, 768, 4, grid=grid(768), stream=stream0)
        buf250 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf245, buf244, buf240, buf250, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf251 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (2304, 512), (1, 2304), 0), view_108, out=buf251)
        del view_108
        buf252 = reinterpret_tensor(buf245, (512, 768), (768, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (512, 2304), (2304, 1), 0), permute_298, out=buf252)
        del permute_298
        buf253 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf254 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_89], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf234, buf252, sub_49, sqrt_18, buf253, buf254, 768, 512, grid=grid(768), stream=stream0)
        buf258 = reinterpret_tensor(buf244, (1, 512, 768), (393216, 768, 1), 0); del buf244  # reuse
        buf259 = reinterpret_tensor(buf240, (1, 512, 768), (393216, 768, 1), 0); del buf240  # reuse
        # Source Nodes: [hidden_states_89, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf234, buf252, primals_37, sub_49, sqrt_18, convert_element_type_24, buf258, buf259, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_24
        del primals_37
        del sqrt_18
        del sub_49
        buf260 = reinterpret_tensor(buf224, (512, 3072), (3072, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (512, 768), (768, 1), 0), permute_300, out=buf260)
        del permute_300
        buf261 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (768, 512), (1, 768), 0), view_106, out=buf261)
        del view_106
        buf262 = reinterpret_tensor(buf248, (1, 768, 4), (3072, 1, 768), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf259, buf262, 3072, 128, grid=grid(3072), stream=stream0)
        buf263 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf262, buf263, 768, 4, grid=grid(768), stream=stream0)
        buf264 = reinterpret_tensor(buf260, (1, 512, 3072), (1572864, 3072, 1), 0); del buf260  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf264, addmm_16, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_16
        buf265 = reinterpret_tensor(buf259, (512, 768), (768, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (512, 3072), (3072, 1), 0), permute_304, out=buf265)
        del permute_304
        buf266 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (3072, 512), (1, 3072), 0), view_104, out=buf266)
        del view_104
        buf267 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf264, buf267, 12288, 128, grid=grid(12288), stream=stream0)
        buf268 = reinterpret_tensor(buf262, (1, 3072), (3072, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf267, buf268, 3072, 4, grid=grid(3072), stream=stream0)
        buf269 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf270 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_81], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf258, buf265, sub_46, sqrt_17, buf269, buf270, 768, 512, grid=grid(768), stream=stream0)
        buf274 = reinterpret_tensor(buf252, (1, 512, 768), (393216, 768, 1), 0); del buf252  # reuse
        buf275 = buf234; del buf234  # reuse
        # Source Nodes: [hidden_states_81, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf258, buf265, primals_35, sub_46, sqrt_17, convert_element_type_23, buf274, buf275, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_23
        del primals_35
        del sqrt_17
        del sub_46
        buf276 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (512, 768), (768, 1), 0), permute_308, out=buf276)
        del permute_308
        buf277 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (768, 512), (1, 768), 0), view_102, out=buf277)
        del view_102
        buf278 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf275, buf278, 3072, 128, grid=grid(3072), stream=stream0)
        buf279 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf278, buf279, 768, 4, grid=grid(768), stream=stream0)
        buf280 = reinterpret_tensor(buf275, (12, 512, 64), (32768, 64, 1), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_313, reinterpret_tensor(buf276, (12, 512, 64), (64, 768, 1), 0), out=buf280)
        del permute_313
        buf281 = reinterpret_tensor(buf243, (12, 512, 512), (262144, 512, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf276, (12, 512, 64), (64, 768, 1), 0), permute_314, out=buf281)
        del permute_314
        buf283 = reinterpret_tensor(buf241, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf241  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_22, buf281, alias_75, buf283, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_75
        del convert_element_type_22
        buf284 = reinterpret_tensor(buf276, (12, 64, 512), (32768, 512, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_315, reinterpret_tensor(buf283, (12, 512, 512), (262144, 512, 1), 0), out=buf284)
        del permute_315
        buf285 = reinterpret_tensor(buf258, (12, 512, 64), (32768, 64, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf283, (12, 512, 512), (262144, 512, 1), 0), permute_316, out=buf285)
        del permute_316
        buf286 = reinterpret_tensor(buf278, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf280, buf286, 3072, 128, grid=grid(3072), stream=stream0)
        buf287 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf286, buf287, 768, 4, grid=grid(768), stream=stream0)
        buf288 = buf286; del buf286  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf285, buf288, 3072, 128, grid=grid(3072), stream=stream0)
        buf289 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf288, buf289, 768, 4, grid=grid(768), stream=stream0)
        buf290 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf285, buf284, buf280, buf290, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf291 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (2304, 512), (1, 2304), 0), view_90, out=buf291)
        del view_90
        buf292 = reinterpret_tensor(buf285, (512, 768), (768, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (512, 2304), (2304, 1), 0), permute_323, out=buf292)
        del permute_323
        buf293 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf294 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_74], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf274, buf292, sub_41, sqrt_15, buf293, buf294, 768, 512, grid=grid(768), stream=stream0)
        buf298 = reinterpret_tensor(buf284, (1, 512, 768), (393216, 768, 1), 0); del buf284  # reuse
        buf299 = reinterpret_tensor(buf280, (1, 512, 768), (393216, 768, 1), 0); del buf280  # reuse
        # Source Nodes: [hidden_states_74, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf274, buf292, primals_31, sub_41, sqrt_15, convert_element_type_20, buf298, buf299, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_20
        del primals_31
        del sqrt_15
        del sub_41
        buf300 = reinterpret_tensor(buf264, (512, 3072), (3072, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (512, 768), (768, 1), 0), permute_325, out=buf300)
        del permute_325
        buf301 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (768, 512), (1, 768), 0), view_88, out=buf301)
        del view_88
        buf302 = reinterpret_tensor(buf288, (1, 768, 4), (3072, 1, 768), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf299, buf302, 3072, 128, grid=grid(3072), stream=stream0)
        buf303 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf302, buf303, 768, 4, grid=grid(768), stream=stream0)
        buf304 = reinterpret_tensor(buf300, (1, 512, 3072), (1572864, 3072, 1), 0); del buf300  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf304, addmm_13, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_13
        buf305 = reinterpret_tensor(buf299, (512, 768), (768, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (512, 3072), (3072, 1), 0), permute_329, out=buf305)
        del permute_329
        buf306 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (3072, 512), (1, 3072), 0), view_86, out=buf306)
        del view_86
        buf307 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf304, buf307, 12288, 128, grid=grid(12288), stream=stream0)
        buf308 = reinterpret_tensor(buf302, (1, 3072), (3072, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf307, buf308, 3072, 4, grid=grid(3072), stream=stream0)
        buf309 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf310 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf298, buf305, sub_38, sqrt_14, buf309, buf310, 768, 512, grid=grid(768), stream=stream0)
        buf314 = reinterpret_tensor(buf292, (1, 512, 768), (393216, 768, 1), 0); del buf292  # reuse
        buf315 = buf274; del buf274  # reuse
        # Source Nodes: [hidden_states_66, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf298, buf305, primals_29, sub_38, sqrt_14, convert_element_type_19, buf314, buf315, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_19
        del primals_29
        del sqrt_14
        del sub_38
        buf316 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (512, 768), (768, 1), 0), permute_333, out=buf316)
        del permute_333
        buf317 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (768, 512), (1, 768), 0), view_84, out=buf317)
        del view_84
        buf318 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf315, buf318, 3072, 128, grid=grid(3072), stream=stream0)
        buf319 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf318, buf319, 768, 4, grid=grid(768), stream=stream0)
        buf320 = reinterpret_tensor(buf315, (12, 512, 64), (32768, 64, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_338, reinterpret_tensor(buf316, (12, 512, 64), (64, 768, 1), 0), out=buf320)
        del permute_338
        buf321 = reinterpret_tensor(buf283, (12, 512, 512), (262144, 512, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf316, (12, 512, 64), (64, 768, 1), 0), permute_339, out=buf321)
        del permute_339
        buf323 = reinterpret_tensor(buf281, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf281  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_18, buf321, alias_80, buf323, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_80
        del convert_element_type_18
        buf324 = reinterpret_tensor(buf316, (12, 64, 512), (32768, 512, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_340, reinterpret_tensor(buf323, (12, 512, 512), (262144, 512, 1), 0), out=buf324)
        del permute_340
        buf325 = reinterpret_tensor(buf298, (12, 512, 64), (32768, 64, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf323, (12, 512, 512), (262144, 512, 1), 0), permute_341, out=buf325)
        del permute_341
        buf326 = reinterpret_tensor(buf318, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf320, buf326, 3072, 128, grid=grid(3072), stream=stream0)
        buf327 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf326, buf327, 768, 4, grid=grid(768), stream=stream0)
        buf328 = buf326; del buf326  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf325, buf328, 3072, 128, grid=grid(3072), stream=stream0)
        buf329 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf328, buf329, 768, 4, grid=grid(768), stream=stream0)
        buf330 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf325, buf324, buf320, buf330, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf331 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (2304, 512), (1, 2304), 0), view_72, out=buf331)
        del view_72
        buf332 = reinterpret_tensor(buf325, (512, 768), (768, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (512, 2304), (2304, 1), 0), permute_348, out=buf332)
        del permute_348
        buf333 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf334 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_59], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf314, buf332, sub_33, sqrt_12, buf333, buf334, 768, 512, grid=grid(768), stream=stream0)
        buf338 = reinterpret_tensor(buf324, (1, 512, 768), (393216, 768, 1), 0); del buf324  # reuse
        buf339 = reinterpret_tensor(buf320, (1, 512, 768), (393216, 768, 1), 0); del buf320  # reuse
        # Source Nodes: [hidden_states_59, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf314, buf332, primals_25, sub_33, sqrt_12, convert_element_type_16, buf338, buf339, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_16
        del primals_25
        del sqrt_12
        del sub_33
        buf340 = reinterpret_tensor(buf304, (512, 3072), (3072, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (512, 768), (768, 1), 0), permute_350, out=buf340)
        del permute_350
        buf341 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (768, 512), (1, 768), 0), view_70, out=buf341)
        del view_70
        buf342 = reinterpret_tensor(buf328, (1, 768, 4), (3072, 1, 768), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf339, buf342, 3072, 128, grid=grid(3072), stream=stream0)
        buf343 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf342, buf343, 768, 4, grid=grid(768), stream=stream0)
        buf344 = reinterpret_tensor(buf340, (1, 512, 3072), (1572864, 3072, 1), 0); del buf340  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf344, addmm_10, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_10
        buf345 = reinterpret_tensor(buf339, (512, 768), (768, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (512, 3072), (3072, 1), 0), permute_354, out=buf345)
        del permute_354
        buf346 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (3072, 512), (1, 3072), 0), view_68, out=buf346)
        del view_68
        buf347 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf344, buf347, 12288, 128, grid=grid(12288), stream=stream0)
        buf348 = reinterpret_tensor(buf342, (1, 3072), (3072, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf347, buf348, 3072, 4, grid=grid(3072), stream=stream0)
        buf349 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf350 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_51], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf338, buf345, sub_30, sqrt_11, buf349, buf350, 768, 512, grid=grid(768), stream=stream0)
        buf354 = reinterpret_tensor(buf332, (1, 512, 768), (393216, 768, 1), 0); del buf332  # reuse
        buf355 = buf314; del buf314  # reuse
        # Source Nodes: [hidden_states_51, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf338, buf345, primals_23, sub_30, sqrt_11, convert_element_type_15, buf354, buf355, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_15
        del primals_23
        del sqrt_11
        del sub_30
        buf356 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (512, 768), (768, 1), 0), permute_358, out=buf356)
        del permute_358
        buf357 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (768, 512), (1, 768), 0), view_66, out=buf357)
        del view_66
        buf358 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf355, buf358, 3072, 128, grid=grid(3072), stream=stream0)
        buf359 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf358, buf359, 768, 4, grid=grid(768), stream=stream0)
        buf360 = reinterpret_tensor(buf355, (12, 512, 64), (32768, 64, 1), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_363, reinterpret_tensor(buf356, (12, 512, 64), (64, 768, 1), 0), out=buf360)
        del permute_363
        buf361 = reinterpret_tensor(buf323, (12, 512, 512), (262144, 512, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf356, (12, 512, 64), (64, 768, 1), 0), permute_364, out=buf361)
        del permute_364
        buf363 = reinterpret_tensor(buf321, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf321  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_14, buf361, alias_85, buf363, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_85
        del convert_element_type_14
        buf364 = reinterpret_tensor(buf356, (12, 64, 512), (32768, 512, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_365, reinterpret_tensor(buf363, (12, 512, 512), (262144, 512, 1), 0), out=buf364)
        del permute_365
        buf365 = reinterpret_tensor(buf338, (12, 512, 64), (32768, 64, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf363, (12, 512, 512), (262144, 512, 1), 0), permute_366, out=buf365)
        del permute_366
        buf366 = reinterpret_tensor(buf358, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf360, buf366, 3072, 128, grid=grid(3072), stream=stream0)
        buf367 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf366, buf367, 768, 4, grid=grid(768), stream=stream0)
        buf368 = buf366; del buf366  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf365, buf368, 3072, 128, grid=grid(3072), stream=stream0)
        buf369 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf368, buf369, 768, 4, grid=grid(768), stream=stream0)
        buf370 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf365, buf364, buf360, buf370, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf371 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (2304, 512), (1, 2304), 0), view_54, out=buf371)
        del view_54
        buf372 = reinterpret_tensor(buf365, (512, 768), (768, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (512, 2304), (2304, 1), 0), permute_373, out=buf372)
        del permute_373
        buf373 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf374 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_44], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf354, buf372, sub_25, sqrt_9, buf373, buf374, 768, 512, grid=grid(768), stream=stream0)
        buf378 = reinterpret_tensor(buf364, (1, 512, 768), (393216, 768, 1), 0); del buf364  # reuse
        buf379 = reinterpret_tensor(buf360, (1, 512, 768), (393216, 768, 1), 0); del buf360  # reuse
        # Source Nodes: [hidden_states_44, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf354, buf372, primals_19, sub_25, sqrt_9, convert_element_type_12, buf378, buf379, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_12
        del primals_19
        del sqrt_9
        del sub_25
        buf380 = reinterpret_tensor(buf344, (512, 3072), (3072, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf379, (512, 768), (768, 1), 0), permute_375, out=buf380)
        del permute_375
        buf381 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf379, (768, 512), (1, 768), 0), view_52, out=buf381)
        del view_52
        buf382 = reinterpret_tensor(buf368, (1, 768, 4), (3072, 1, 768), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf379, buf382, 3072, 128, grid=grid(3072), stream=stream0)
        buf383 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf382, buf383, 768, 4, grid=grid(768), stream=stream0)
        buf384 = reinterpret_tensor(buf380, (1, 512, 3072), (1572864, 3072, 1), 0); del buf380  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf384, addmm_7, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_7
        buf385 = reinterpret_tensor(buf379, (512, 768), (768, 1), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (512, 3072), (3072, 1), 0), permute_379, out=buf385)
        del permute_379
        buf386 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (3072, 512), (1, 3072), 0), view_50, out=buf386)
        del view_50
        buf387 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf384, buf387, 12288, 128, grid=grid(12288), stream=stream0)
        buf388 = reinterpret_tensor(buf382, (1, 3072), (3072, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf387, buf388, 3072, 4, grid=grid(3072), stream=stream0)
        buf389 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf390 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf378, buf385, sub_22, sqrt_8, buf389, buf390, 768, 512, grid=grid(768), stream=stream0)
        buf394 = reinterpret_tensor(buf372, (1, 512, 768), (393216, 768, 1), 0); del buf372  # reuse
        buf395 = buf354; del buf354  # reuse
        # Source Nodes: [hidden_states_36, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf378, buf385, primals_17, sub_22, sqrt_8, convert_element_type_11, buf394, buf395, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_11
        del primals_17
        del sqrt_8
        del sub_22
        buf396 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (512, 768), (768, 1), 0), permute_383, out=buf396)
        del permute_383
        buf397 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (768, 512), (1, 768), 0), view_48, out=buf397)
        del view_48
        buf398 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf395, buf398, 3072, 128, grid=grid(3072), stream=stream0)
        buf399 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf398, buf399, 768, 4, grid=grid(768), stream=stream0)
        buf400 = reinterpret_tensor(buf395, (12, 512, 64), (32768, 64, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_388, reinterpret_tensor(buf396, (12, 512, 64), (64, 768, 1), 0), out=buf400)
        del permute_388
        buf401 = reinterpret_tensor(buf363, (12, 512, 512), (262144, 512, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf396, (12, 512, 64), (64, 768, 1), 0), permute_389, out=buf401)
        del permute_389
        buf403 = reinterpret_tensor(buf361, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf361  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_10, buf401, alias_90, buf403, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_90
        del convert_element_type_10
        buf404 = reinterpret_tensor(buf396, (12, 64, 512), (32768, 512, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_390, reinterpret_tensor(buf403, (12, 512, 512), (262144, 512, 1), 0), out=buf404)
        del permute_390
        buf405 = reinterpret_tensor(buf378, (12, 512, 64), (32768, 64, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf403, (12, 512, 512), (262144, 512, 1), 0), permute_391, out=buf405)
        del permute_391
        buf406 = reinterpret_tensor(buf398, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf400, buf406, 3072, 128, grid=grid(3072), stream=stream0)
        buf407 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf406, buf407, 768, 4, grid=grid(768), stream=stream0)
        buf408 = buf406; del buf406  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf405, buf408, 3072, 128, grid=grid(3072), stream=stream0)
        buf409 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf408, buf409, 768, 4, grid=grid(768), stream=stream0)
        buf410 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf405, buf404, buf400, buf410, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf411 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (2304, 512), (1, 2304), 0), view_36, out=buf411)
        del view_36
        buf412 = reinterpret_tensor(buf405, (512, 768), (768, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (512, 2304), (2304, 1), 0), permute_398, out=buf412)
        del permute_398
        buf413 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf414 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_29], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf394, buf412, sub_17, sqrt_6, buf413, buf414, 768, 512, grid=grid(768), stream=stream0)
        buf418 = reinterpret_tensor(buf404, (1, 512, 768), (393216, 768, 1), 0); del buf404  # reuse
        buf419 = reinterpret_tensor(buf400, (1, 512, 768), (393216, 768, 1), 0); del buf400  # reuse
        # Source Nodes: [hidden_states_29, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf394, buf412, primals_13, sub_17, sqrt_6, convert_element_type_8, buf418, buf419, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_8
        del primals_13
        del sqrt_6
        del sub_17
        buf420 = reinterpret_tensor(buf384, (512, 3072), (3072, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (512, 768), (768, 1), 0), permute_400, out=buf420)
        del permute_400
        buf421 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (768, 512), (1, 768), 0), view_34, out=buf421)
        del view_34
        buf422 = reinterpret_tensor(buf408, (1, 768, 4), (3072, 1, 768), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf419, buf422, 3072, 128, grid=grid(3072), stream=stream0)
        buf423 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf422, buf423, 768, 4, grid=grid(768), stream=stream0)
        buf424 = reinterpret_tensor(buf420, (1, 512, 3072), (1572864, 3072, 1), 0); del buf420  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf424, addmm_4, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_4
        buf425 = reinterpret_tensor(buf419, (512, 768), (768, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (512, 3072), (3072, 1), 0), permute_404, out=buf425)
        del permute_404
        buf426 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (3072, 512), (1, 3072), 0), view_32, out=buf426)
        del view_32
        buf427 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf424, buf427, 12288, 128, grid=grid(12288), stream=stream0)
        buf428 = reinterpret_tensor(buf422, (1, 3072), (3072, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf427, buf428, 3072, 4, grid=grid(3072), stream=stream0)
        buf429 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf430 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_21], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf418, buf425, sub_14, sqrt_5, buf429, buf430, 768, 512, grid=grid(768), stream=stream0)
        buf434 = reinterpret_tensor(buf412, (1, 512, 768), (393216, 768, 1), 0); del buf412  # reuse
        buf435 = buf394; del buf394  # reuse
        # Source Nodes: [hidden_states_21, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf418, buf425, primals_11, sub_14, sqrt_5, convert_element_type_7, buf434, buf435, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_7
        del primals_11
        del sqrt_5
        del sub_14
        buf436 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (512, 768), (768, 1), 0), permute_408, out=buf436)
        del permute_408
        buf437 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (768, 512), (1, 768), 0), view_30, out=buf437)
        del view_30
        buf438 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf435, buf438, 3072, 128, grid=grid(3072), stream=stream0)
        buf439 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf438, buf439, 768, 4, grid=grid(768), stream=stream0)
        buf440 = reinterpret_tensor(buf435, (12, 512, 64), (32768, 64, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_413, reinterpret_tensor(buf436, (12, 512, 64), (64, 768, 1), 0), out=buf440)
        del permute_413
        buf441 = reinterpret_tensor(buf403, (12, 512, 512), (262144, 512, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf436, (12, 512, 64), (64, 768, 1), 0), permute_414, out=buf441)
        del permute_414
        buf443 = reinterpret_tensor(buf401, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf401  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_6, buf441, alias_95, buf443, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_95
        del convert_element_type_6
        buf444 = reinterpret_tensor(buf436, (12, 64, 512), (32768, 512, 1), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_415, reinterpret_tensor(buf443, (12, 512, 512), (262144, 512, 1), 0), out=buf444)
        del permute_415
        buf445 = reinterpret_tensor(buf418, (12, 512, 64), (32768, 64, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf443, (12, 512, 512), (262144, 512, 1), 0), permute_416, out=buf445)
        del permute_416
        buf446 = reinterpret_tensor(buf438, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf440, buf446, 3072, 128, grid=grid(3072), stream=stream0)
        buf447 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf446, buf447, 768, 4, grid=grid(768), stream=stream0)
        buf448 = buf446; del buf446  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf445, buf448, 3072, 128, grid=grid(3072), stream=stream0)
        buf449 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf448, buf449, 768, 4, grid=grid(768), stream=stream0)
        buf450 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf445, buf444, buf440, buf450, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf451 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (2304, 512), (1, 2304), 0), view_18, out=buf451)
        del view_18
        buf452 = reinterpret_tensor(buf445, (512, 768), (768, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (512, 2304), (2304, 1), 0), permute_423, out=buf452)
        del permute_423
        buf453 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf454 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf434, buf452, sub_9, sqrt_3, buf453, buf454, 768, 512, grid=grid(768), stream=stream0)
        buf458 = reinterpret_tensor(buf444, (1, 512, 768), (393216, 768, 1), 0); del buf444  # reuse
        buf459 = reinterpret_tensor(buf440, (1, 512, 768), (393216, 768, 1), 0); del buf440  # reuse
        # Source Nodes: [hidden_states_14, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf434, buf452, primals_7, sub_9, sqrt_3, convert_element_type_4, buf458, buf459, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_4
        del primals_7
        del sqrt_3
        del sub_9
        buf460 = reinterpret_tensor(buf424, (512, 3072), (3072, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (512, 768), (768, 1), 0), permute_425, out=buf460)
        del permute_425
        buf461 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (768, 512), (1, 768), 0), view_16, out=buf461)
        del view_16
        buf462 = reinterpret_tensor(buf448, (1, 768, 4), (3072, 1, 768), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf459, buf462, 3072, 128, grid=grid(3072), stream=stream0)
        buf463 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf462, buf463, 768, 4, grid=grid(768), stream=stream0)
        buf464 = reinterpret_tensor(buf460, (1, 512, 3072), (1572864, 3072, 1), 0); del buf460  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf464, addmm_1, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_1
        buf465 = reinterpret_tensor(buf459, (512, 768), (768, 1), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (512, 3072), (3072, 1), 0), permute_429, out=buf465)
        del permute_429
        buf466 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (3072, 512), (1, 3072), 0), view_14, out=buf466)
        del view_14
        buf467 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf464, buf467, 12288, 128, grid=grid(12288), stream=stream0)
        del buf464
        buf468 = reinterpret_tensor(buf462, (1, 3072), (3072, 1), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf467, buf468, 3072, 4, grid=grid(3072), stream=stream0)
        del buf467
        buf469 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf470 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_6], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_12.run(buf458, buf465, sub_6, sqrt_2, buf469, buf470, 768, 512, grid=grid(768), stream=stream0)
        buf474 = reinterpret_tensor(buf452, (1, 512, 768), (393216, 768, 1), 0); del buf452  # reuse
        buf475 = buf434; del buf434  # reuse
        # Source Nodes: [hidden_states_6, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_13.run(buf458, buf465, primals_5, sub_6, sqrt_2, convert_element_type_3, buf474, buf475, 512, 768, grid=grid(512), stream=stream0)
        del convert_element_type_3
        del primals_5
        del sqrt_2
        del sub_6
        buf476 = buf465; del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (512, 768), (768, 1), 0), permute_433, out=buf476)
        del permute_433
        buf477 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (768, 512), (1, 768), 0), view_12, out=buf477)
        del view_12
        buf478 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf475, buf478, 3072, 128, grid=grid(3072), stream=stream0)
        buf479 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf478, buf479, 768, 4, grid=grid(768), stream=stream0)
        buf480 = reinterpret_tensor(buf475, (12, 512, 64), (32768, 64, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_438, reinterpret_tensor(buf476, (12, 512, 64), (64, 768, 1), 0), out=buf480)
        del permute_438
        buf481 = reinterpret_tensor(buf443, (12, 512, 512), (262144, 512, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf476, (12, 512, 64), (64, 768, 1), 0), permute_439, out=buf481)
        del permute_439
        buf483 = reinterpret_tensor(buf441, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf441  # reuse
        # Source Nodes: [query_states], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.mul]
        triton_per_fused__softmax_backward_data_masked_fill_mul_14.run(convert_element_type_2, buf481, alias_100, buf483, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_100
        del buf481
        del convert_element_type_2
        buf484 = reinterpret_tensor(buf476, (12, 64, 512), (32768, 512, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_440, reinterpret_tensor(buf483, (12, 512, 512), (262144, 512, 1), 0), out=buf484)
        del permute_440
        buf485 = reinterpret_tensor(buf458, (12, 512, 64), (32768, 64, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf483, (12, 512, 512), (262144, 512, 1), 0), permute_441, out=buf485)
        del buf483
        del permute_441
        buf486 = reinterpret_tensor(buf478, (1, 12, 1, 64, 4), (3072, 256, 3072, 1, 64), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf480, buf486, 3072, 128, grid=grid(3072), stream=stream0)
        buf487 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf486, buf487, 768, 4, grid=grid(768), stream=stream0)
        buf488 = buf486; del buf486  # reuse
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_red_fused_div_sqrt_sum_17.run(buf485, buf488, 3072, 128, grid=grid(3072), stream=stream0)
        buf489 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [scale], Original ATen: [aten.div, aten.sqrt, aten.sum]
        triton_per_fused_sum_16.run(buf488, buf489, 768, 4, grid=grid(768), stream=stream0)
        del buf488
        buf490 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf485, buf484, buf480, buf490, 6144, 192, grid=grid(6144, 192), stream=stream0)
        buf491 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (2304, 512), (1, 2304), 0), view, out=buf491)
        del view
        buf492 = reinterpret_tensor(buf485, (512, 768), (768, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (512, 2304), (2304, 1), 0), permute_448, out=buf492)
        del buf490
        del permute_448
        buf493 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        buf494 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_1, query_states], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.sum]
        triton_per_fused_add_div_masked_fill_mul_sum_19.run(convert_element_type, buf474, buf492, sub, sqrt, buf493, buf494, 768, 512, grid=grid(768), stream=stream0)
        buf501 = reinterpret_tensor(buf484, (1, 512, 768), (393216, 768, 1), 0); del buf484  # reuse
        buf505 = reinterpret_tensor(buf480, (1, 512, 768), (393216, 768, 1), 0); del buf480  # reuse
        # Source Nodes: [hidden_states_1, query_states], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
        triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_20.run(convert_element_type, buf474, buf492, primals_1, sqrt, sub, slice_1, primals_164, buf501, buf505, 512, 768, grid=grid(512), stream=stream0)
        del buf474
        del convert_element_type
        del primals_1
        del sqrt
        del sub
        buf500 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_21.run(buf500, 393216, grid=grid(393216), stream=stream0)
        aten.index_put_(buf500, [slice_1], buf501, True)
        del buf501
        del slice_1
        buf504 = empty((50265, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_22.run(buf504, 38603520, grid=grid(38603520), stream=stream0)
        aten.index_put_(buf504, [primals_164], buf505, True)
        del buf505
        del primals_164
        return (reinterpret_tensor(buf494, (768, ), (1, ), 0), reinterpret_tensor(buf493, (768, ), (1, ), 0), reinterpret_tensor(buf489, (768, ), (1, ), 0), reinterpret_tensor(buf487, (768, ), (1, ), 0), reinterpret_tensor(buf470, (768, ), (1, ), 0), reinterpret_tensor(buf469, (768, ), (1, ), 0), reinterpret_tensor(buf454, (768, ), (1, ), 0), reinterpret_tensor(buf453, (768, ), (1, ), 0), reinterpret_tensor(buf449, (768, ), (1, ), 0), reinterpret_tensor(buf447, (768, ), (1, ), 0), reinterpret_tensor(buf430, (768, ), (1, ), 0), reinterpret_tensor(buf429, (768, ), (1, ), 0), reinterpret_tensor(buf414, (768, ), (1, ), 0), reinterpret_tensor(buf413, (768, ), (1, ), 0), reinterpret_tensor(buf409, (768, ), (1, ), 0), reinterpret_tensor(buf407, (768, ), (1, ), 0), reinterpret_tensor(buf390, (768, ), (1, ), 0), reinterpret_tensor(buf389, (768, ), (1, ), 0), reinterpret_tensor(buf374, (768, ), (1, ), 0), reinterpret_tensor(buf373, (768, ), (1, ), 0), reinterpret_tensor(buf369, (768, ), (1, ), 0), reinterpret_tensor(buf367, (768, ), (1, ), 0), reinterpret_tensor(buf350, (768, ), (1, ), 0), reinterpret_tensor(buf349, (768, ), (1, ), 0), reinterpret_tensor(buf334, (768, ), (1, ), 0), reinterpret_tensor(buf333, (768, ), (1, ), 0), reinterpret_tensor(buf329, (768, ), (1, ), 0), reinterpret_tensor(buf327, (768, ), (1, ), 0), reinterpret_tensor(buf310, (768, ), (1, ), 0), reinterpret_tensor(buf309, (768, ), (1, ), 0), reinterpret_tensor(buf294, (768, ), (1, ), 0), reinterpret_tensor(buf293, (768, ), (1, ), 0), reinterpret_tensor(buf289, (768, ), (1, ), 0), reinterpret_tensor(buf287, (768, ), (1, ), 0), reinterpret_tensor(buf270, (768, ), (1, ), 0), reinterpret_tensor(buf269, (768, ), (1, ), 0), reinterpret_tensor(buf254, (768, ), (1, ), 0), reinterpret_tensor(buf253, (768, ), (1, ), 0), reinterpret_tensor(buf249, (768, ), (1, ), 0), reinterpret_tensor(buf247, (768, ), (1, ), 0), reinterpret_tensor(buf230, (768, ), (1, ), 0), reinterpret_tensor(buf229, (768, ), (1, ), 0), reinterpret_tensor(buf214, (768, ), (1, ), 0), reinterpret_tensor(buf213, (768, ), (1, ), 0), reinterpret_tensor(buf209, (768, ), (1, ), 0), reinterpret_tensor(buf207, (768, ), (1, ), 0), reinterpret_tensor(buf190, (768, ), (1, ), 0), reinterpret_tensor(buf189, (768, ), (1, ), 0), reinterpret_tensor(buf174, (768, ), (1, ), 0), reinterpret_tensor(buf173, (768, ), (1, ), 0), reinterpret_tensor(buf169, (768, ), (1, ), 0), reinterpret_tensor(buf167, (768, ), (1, ), 0), reinterpret_tensor(buf150, (768, ), (1, ), 0), reinterpret_tensor(buf149, (768, ), (1, ), 0), reinterpret_tensor(buf134, (768, ), (1, ), 0), reinterpret_tensor(buf133, (768, ), (1, ), 0), reinterpret_tensor(buf129, (768, ), (1, ), 0), reinterpret_tensor(buf127, (768, ), (1, ), 0), reinterpret_tensor(buf110, (768, ), (1, ), 0), reinterpret_tensor(buf109, (768, ), (1, ), 0), reinterpret_tensor(buf94, (768, ), (1, ), 0), reinterpret_tensor(buf93, (768, ), (1, ), 0), reinterpret_tensor(buf89, (768, ), (1, ), 0), reinterpret_tensor(buf87, (768, ), (1, ), 0), reinterpret_tensor(buf70, (768, ), (1, ), 0), reinterpret_tensor(buf69, (768, ), (1, ), 0), reinterpret_tensor(buf54, (768, ), (1, ), 0), reinterpret_tensor(buf53, (768, ), (1, ), 0), reinterpret_tensor(buf49, (768, ), (1, ), 0), reinterpret_tensor(buf47, (768, ), (1, ), 0), reinterpret_tensor(buf30, (768, ), (1, ), 0), reinterpret_tensor(buf29, (768, ), (1, ), 0), reinterpret_tensor(buf14, (768, ), (1, ), 0), reinterpret_tensor(buf13, (768, ), (1, ), 0), buf504, buf500, reinterpret_tensor(buf491, (2304, 768), (768, 1), 0), reinterpret_tensor(buf477, (768, 768), (768, 1), 0), reinterpret_tensor(buf479, (768, ), (1, ), 0), reinterpret_tensor(buf466, (3072, 768), (768, 1), 0), reinterpret_tensor(buf468, (3072, ), (1, ), 0), reinterpret_tensor(buf461, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf463, (768, ), (1, ), 0), reinterpret_tensor(buf451, (2304, 768), (768, 1), 0), reinterpret_tensor(buf437, (768, 768), (768, 1), 0), reinterpret_tensor(buf439, (768, ), (1, ), 0), reinterpret_tensor(buf426, (3072, 768), (768, 1), 0), reinterpret_tensor(buf428, (3072, ), (1, ), 0), reinterpret_tensor(buf421, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf423, (768, ), (1, ), 0), reinterpret_tensor(buf411, (2304, 768), (768, 1), 0), reinterpret_tensor(buf397, (768, 768), (768, 1), 0), reinterpret_tensor(buf399, (768, ), (1, ), 0), reinterpret_tensor(buf386, (3072, 768), (768, 1), 0), reinterpret_tensor(buf388, (3072, ), (1, ), 0), reinterpret_tensor(buf381, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf383, (768, ), (1, ), 0), reinterpret_tensor(buf371, (2304, 768), (768, 1), 0), reinterpret_tensor(buf357, (768, 768), (768, 1), 0), reinterpret_tensor(buf359, (768, ), (1, ), 0), reinterpret_tensor(buf346, (3072, 768), (768, 1), 0), reinterpret_tensor(buf348, (3072, ), (1, ), 0), reinterpret_tensor(buf341, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf343, (768, ), (1, ), 0), reinterpret_tensor(buf331, (2304, 768), (768, 1), 0), reinterpret_tensor(buf317, (768, 768), (768, 1), 0), reinterpret_tensor(buf319, (768, ), (1, ), 0), reinterpret_tensor(buf306, (3072, 768), (768, 1), 0), reinterpret_tensor(buf308, (3072, ), (1, ), 0), reinterpret_tensor(buf301, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf303, (768, ), (1, ), 0), reinterpret_tensor(buf291, (2304, 768), (768, 1), 0), reinterpret_tensor(buf277, (768, 768), (768, 1), 0), reinterpret_tensor(buf279, (768, ), (1, ), 0), reinterpret_tensor(buf266, (3072, 768), (768, 1), 0), reinterpret_tensor(buf268, (3072, ), (1, ), 0), reinterpret_tensor(buf261, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf263, (768, ), (1, ), 0), reinterpret_tensor(buf251, (2304, 768), (768, 1), 0), reinterpret_tensor(buf237, (768, 768), (768, 1), 0), reinterpret_tensor(buf239, (768, ), (1, ), 0), reinterpret_tensor(buf226, (3072, 768), (768, 1), 0), reinterpret_tensor(buf228, (3072, ), (1, ), 0), reinterpret_tensor(buf221, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf223, (768, ), (1, ), 0), reinterpret_tensor(buf211, (2304, 768), (768, 1), 0), reinterpret_tensor(buf197, (768, 768), (768, 1), 0), reinterpret_tensor(buf199, (768, ), (1, ), 0), reinterpret_tensor(buf186, (3072, 768), (768, 1), 0), reinterpret_tensor(buf188, (3072, ), (1, ), 0), reinterpret_tensor(buf181, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf183, (768, ), (1, ), 0), reinterpret_tensor(buf171, (2304, 768), (768, 1), 0), reinterpret_tensor(buf157, (768, 768), (768, 1), 0), reinterpret_tensor(buf159, (768, ), (1, ), 0), reinterpret_tensor(buf146, (3072, 768), (768, 1), 0), reinterpret_tensor(buf148, (3072, ), (1, ), 0), reinterpret_tensor(buf141, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf143, (768, ), (1, ), 0), reinterpret_tensor(buf131, (2304, 768), (768, 1), 0), reinterpret_tensor(buf117, (768, 768), (768, 1), 0), reinterpret_tensor(buf119, (768, ), (1, ), 0), reinterpret_tensor(buf106, (3072, 768), (768, 1), 0), reinterpret_tensor(buf108, (3072, ), (1, ), 0), reinterpret_tensor(buf101, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf103, (768, ), (1, ), 0), reinterpret_tensor(buf91, (2304, 768), (768, 1), 0), reinterpret_tensor(buf77, (768, 768), (768, 1), 0), reinterpret_tensor(buf79, (768, ), (1, ), 0), reinterpret_tensor(buf66, (3072, 768), (768, 1), 0), reinterpret_tensor(buf68, (3072, ), (1, ), 0), reinterpret_tensor(buf61, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf63, (768, ), (1, ), 0), reinterpret_tensor(buf51, (2304, 768), (768, 1), 0), reinterpret_tensor(buf37, (768, 768), (768, 1), 0), reinterpret_tensor(buf39, (768, ), (1, ), 0), reinterpret_tensor(buf26, (3072, 768), (768, 1), 0), reinterpret_tensor(buf28, (3072, ), (1, ), 0), reinterpret_tensor(buf21, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf23, (768, ), (1, ), 0), reinterpret_tensor(buf10, (2, 768), (768, 1), 0), reinterpret_tensor(buf12, (2, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    slice_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    sub = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    view = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_2 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_12 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_6 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_2 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_4 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_9 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_3 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_6 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_30 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_14 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_5 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_6 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_10 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_48 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_22 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_8 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_52 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_12 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_25 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_9 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_54 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_14 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_66 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_15 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_30 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_11 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_16 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_33 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_12 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_18 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_84 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_19 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_38 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_14 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_20 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_15 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_22 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_102 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_23 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_46 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_17 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_49 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_18 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_26 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_120 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_54 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_20 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_122 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_124 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_28 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_21 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_30 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_138 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_62 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_23 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_142 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_32 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_65 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_24 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_144 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_34 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_156 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_35 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_70 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_158 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_25 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_160 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_162 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_38 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_174 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_39 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_78 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_29 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_178 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_40 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_180 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_42 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_192 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_43 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_86 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_32 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_31 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_44 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_89 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_46 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    view_210 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_47 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_94 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_35 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_48 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    sub_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    sqrt_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sub_100 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    sub_102 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_65 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_67 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_146 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_154 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_158 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_164 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_165 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_173 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_188 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_50 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_198 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_213 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_214 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_55 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_215 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_225 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_229 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_239 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_60 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_248 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_263 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_65 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_273 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_275 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_279 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_70 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_298 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_300 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_304 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_308 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_313 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_314 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_75 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_316 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_325 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_329 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_333 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_338 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_339 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_80 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_341 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_358 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_363 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_364 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_85 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_366 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_375 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_379 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_90 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_390 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_391 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_400 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_408 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_413 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_95 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_416 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_423 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_425 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_429 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_433 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_100 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_440 = rand_strided((12, 64, 512), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_441 = rand_strided((12, 512, 64), (192, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_164, slice_1, sub, sqrt, convert_element_type, view, convert_element_type_2, view_12, convert_element_type_3, sub_6, sqrt_2, view_14, addmm_1, view_16, convert_element_type_4, sub_9, sqrt_3, view_18, convert_element_type_6, view_30, convert_element_type_7, sub_14, sqrt_5, view_32, addmm_4, view_34, convert_element_type_8, sub_17, sqrt_6, view_36, convert_element_type_10, view_48, convert_element_type_11, sub_22, sqrt_8, view_50, addmm_7, view_52, convert_element_type_12, sub_25, sqrt_9, view_54, convert_element_type_14, view_66, convert_element_type_15, sub_30, sqrt_11, view_68, addmm_10, view_70, convert_element_type_16, sub_33, sqrt_12, view_72, convert_element_type_18, view_84, convert_element_type_19, sub_38, sqrt_14, view_86, addmm_13, view_88, convert_element_type_20, sub_41, sqrt_15, view_90, convert_element_type_22, view_102, convert_element_type_23, sub_46, sqrt_17, view_104, addmm_16, view_106, convert_element_type_24, sub_49, sqrt_18, view_108, convert_element_type_26, view_120, convert_element_type_27, sub_54, sqrt_20, view_122, addmm_19, view_124, convert_element_type_28, sub_57, sqrt_21, view_126, convert_element_type_30, view_138, convert_element_type_31, sub_62, sqrt_23, view_140, addmm_22, view_142, convert_element_type_32, sub_65, sqrt_24, view_144, convert_element_type_34, view_156, convert_element_type_35, sub_70, sqrt_26, view_158, addmm_25, view_160, convert_element_type_36, sub_73, sqrt_27, view_162, convert_element_type_38, view_174, convert_element_type_39, sub_78, sqrt_29, view_176, addmm_28, view_178, convert_element_type_40, sub_81, sqrt_30, view_180, convert_element_type_42, view_192, convert_element_type_43, sub_86, sqrt_32, view_194, addmm_31, view_196, convert_element_type_44, sub_89, sqrt_33, view_198, convert_element_type_46, view_210, convert_element_type_47, sub_94, sqrt_35, view_212, addmm_34, view_214, convert_element_type_48, sub_97, sqrt_36, view_216, sub_100, ne, sub_102, ne_3, ne_6, where_65, ne_8, where_67, permute_146, permute_150, permute_154, permute_158, permute_163, permute_164, alias_45, permute_165, permute_166, permute_173, permute_175, permute_179, permute_183, permute_188, permute_189, alias_50, permute_190, permute_191, permute_198, permute_200, permute_204, permute_208, permute_213, permute_214, alias_55, permute_215, permute_216, permute_223, permute_225, permute_229, permute_233, permute_238, permute_239, alias_60, permute_240, permute_241, permute_248, permute_250, permute_254, permute_258, permute_263, permute_264, alias_65, permute_265, permute_266, permute_273, permute_275, permute_279, permute_283, permute_288, permute_289, alias_70, permute_290, permute_291, permute_298, permute_300, permute_304, permute_308, permute_313, permute_314, alias_75, permute_315, permute_316, permute_323, permute_325, permute_329, permute_333, permute_338, permute_339, alias_80, permute_340, permute_341, permute_348, permute_350, permute_354, permute_358, permute_363, permute_364, alias_85, permute_365, permute_366, permute_373, permute_375, permute_379, permute_383, permute_388, permute_389, alias_90, permute_390, permute_391, permute_398, permute_400, permute_404, permute_408, permute_413, permute_414, alias_95, permute_415, permute_416, permute_423, permute_425, permute_429, permute_433, permute_438, permute_439, alias_100, permute_440, permute_441, permute_448, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaForQuestionAnswering', benchmark_compiled_module)
