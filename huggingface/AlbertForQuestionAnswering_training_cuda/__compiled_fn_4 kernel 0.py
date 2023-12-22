
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


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtnnqcysyi3smsgan77l277tnkmnwv35xdvvlsgo6vcukvg6ad4.py
# Source Nodes: [end_loss, start_loss], Original ATen: [aten._log_softmax_backward_data, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
# end_loss => convert_element_type_1, sum_17
# start_loss => convert_element_type, full_default_2, sum_14
triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ql/cqldxgwpc3wdwril4fw64sngv73dk6apczkfera6veam4qcupnpl.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tmp12 * tmp13
        tmp15 = 4096.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16 - tmp4
        tmp19 = tmp18 * tmp9
        tmp20 = tmp17 - tmp19
        tmp21 = tmp11 * tmp20
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ik/ciko3xar5fwe2mclf456hi4dsl4d7owtyio4ekqx6jdmpzlytard.py
# Source Nodes: [add_59, mul_45], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
# add_59 => add_108
# mul_45 => mul_93
triton_poi_fused_add_mul_pow_tanh_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr1 + (x0), None)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp4 * tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = 0.044715
    tmp13 = tmp11 * tmp12
    tmp14 = tmp1 * tmp1
    tmp15 = 3.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 * tmp16
    tmp18 = tmp11 + tmp17
    tmp19 = tmp5 + tmp7
    tmp20 = tmp0 * tmp19
    tmp21 = tmp20 * tmp2
    tmp22 = tmp18 + tmp21
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yz/cyz7qfcvltthrki3yroegfvievdkzuujh26snkumbqd7yteu3omb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr3 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = 4096.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20 - tmp6
        tmp23 = tmp22 * tmp11
        tmp24 = tmp21 - tmp23
        tmp25 = tmp13 * tmp24
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gw/cgw3gxc6iokw4jnnd5rxu46uib3ezcg4xu3tn2qxc7c65lzmfgbd.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (32768*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ud/cudykjcdawq4asalkywdyckpvqakand4g2rel7o6tg6fzruj53es.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]

triton_per_fused__softmax_backward_data_div_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_div_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 32768
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp10, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5c/c5c2btilflgpkbioquah6lillrycgl7r3375jvevntaw46mswdek.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tl.store(out_ptr0 + (r1 + (4096*x0)), tmp8, rmask & xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr5 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp12 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = 4096.0
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21 - tmp10
        tmp24 = tmp23 * tmp16
        tmp25 = tmp22 - tmp24
        tmp26 = tmp18 * tmp25
        tl.store(out_ptr3 + (r1 + (4096*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vt/cvt47kii3jjbvtk2wbkhlanixu67vfjwfjdbupspmzf6pofhivkr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: 'i32', 60: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(59, 60))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp18 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp29 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp31 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp35 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp45 = tl.load(in_ptr12 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp46 = tl.load(in_ptr13 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp48 = tl.load(in_ptr14 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp50 = tl.load(in_ptr15 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp52 = tl.load(in_ptr16 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp62 = tl.load(in_ptr17 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp63 = tl.load(in_ptr18 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp65 = tl.load(in_ptr19 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp67 = tl.load(in_ptr20 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp69 = tl.load(in_ptr21 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp79 = tl.load(in_ptr22 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp80 = tl.load(in_ptr23 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp82 = tl.load(in_ptr24 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp84 = tl.load(in_ptr25 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp86 = tl.load(in_ptr26 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp96 = tl.load(in_ptr27 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp97 = tl.load(in_ptr28 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp99 = tl.load(in_ptr29 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp101 = tl.load(in_ptr30 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp103 = tl.load(in_ptr31 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp113 = tl.load(in_ptr32 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp114 = tl.load(in_ptr33 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp116 = tl.load(in_ptr34 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp118 = tl.load(in_ptr35 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp120 = tl.load(in_ptr36 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp130 = tl.load(in_ptr37 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp131 = tl.load(in_ptr38 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp133 = tl.load(in_ptr39 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp135 = tl.load(in_ptr40 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp137 = tl.load(in_ptr41 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp147 = tl.load(in_ptr42 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp148 = tl.load(in_ptr43 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp150 = tl.load(in_ptr44 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp152 = tl.load(in_ptr45 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp154 = tl.load(in_ptr46 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp164 = tl.load(in_ptr47 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp165 = tl.load(in_ptr48 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp167 = tl.load(in_ptr49 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp169 = tl.load(in_ptr50 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp171 = tl.load(in_ptr51 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp181 = tl.load(in_ptr52 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp182 = tl.load(in_ptr53 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp184 = tl.load(in_ptr54 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp186 = tl.load(in_ptr55 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp188 = tl.load(in_ptr56 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [RBLOCK])
    tmp39 = tl.where(rmask, tmp37, 0)
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp39, 0))
    tmp41 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp43 = tl.where(rmask, tmp41, 0)
    tmp44 = triton_helpers.promote_to_tensor(tl.sum(tmp43, 0))
    tmp47 = tmp45 + tmp46
    tmp49 = tmp47 + tmp48
    tmp51 = tmp49 + tmp50
    tmp53 = tmp51 * tmp52
    tmp54 = tl.broadcast_to(tmp53, [RBLOCK])
    tmp56 = tl.where(rmask, tmp54, 0)
    tmp57 = triton_helpers.promote_to_tensor(tl.sum(tmp56, 0))
    tmp58 = tl.broadcast_to(tmp51, [RBLOCK])
    tmp60 = tl.where(rmask, tmp58, 0)
    tmp61 = triton_helpers.promote_to_tensor(tl.sum(tmp60, 0))
    tmp64 = tmp62 + tmp63
    tmp66 = tmp64 + tmp65
    tmp68 = tmp66 + tmp67
    tmp70 = tmp68 * tmp69
    tmp71 = tl.broadcast_to(tmp70, [RBLOCK])
    tmp73 = tl.where(rmask, tmp71, 0)
    tmp74 = triton_helpers.promote_to_tensor(tl.sum(tmp73, 0))
    tmp75 = tl.broadcast_to(tmp68, [RBLOCK])
    tmp77 = tl.where(rmask, tmp75, 0)
    tmp78 = triton_helpers.promote_to_tensor(tl.sum(tmp77, 0))
    tmp81 = tmp79 + tmp80
    tmp83 = tmp81 + tmp82
    tmp85 = tmp83 + tmp84
    tmp87 = tmp85 * tmp86
    tmp88 = tl.broadcast_to(tmp87, [RBLOCK])
    tmp90 = tl.where(rmask, tmp88, 0)
    tmp91 = triton_helpers.promote_to_tensor(tl.sum(tmp90, 0))
    tmp92 = tl.broadcast_to(tmp85, [RBLOCK])
    tmp94 = tl.where(rmask, tmp92, 0)
    tmp95 = triton_helpers.promote_to_tensor(tl.sum(tmp94, 0))
    tmp98 = tmp96 + tmp97
    tmp100 = tmp98 + tmp99
    tmp102 = tmp100 + tmp101
    tmp104 = tmp102 * tmp103
    tmp105 = tl.broadcast_to(tmp104, [RBLOCK])
    tmp107 = tl.where(rmask, tmp105, 0)
    tmp108 = triton_helpers.promote_to_tensor(tl.sum(tmp107, 0))
    tmp109 = tl.broadcast_to(tmp102, [RBLOCK])
    tmp111 = tl.where(rmask, tmp109, 0)
    tmp112 = triton_helpers.promote_to_tensor(tl.sum(tmp111, 0))
    tmp115 = tmp113 + tmp114
    tmp117 = tmp115 + tmp116
    tmp119 = tmp117 + tmp118
    tmp121 = tmp119 * tmp120
    tmp122 = tl.broadcast_to(tmp121, [RBLOCK])
    tmp124 = tl.where(rmask, tmp122, 0)
    tmp125 = triton_helpers.promote_to_tensor(tl.sum(tmp124, 0))
    tmp126 = tl.broadcast_to(tmp119, [RBLOCK])
    tmp128 = tl.where(rmask, tmp126, 0)
    tmp129 = triton_helpers.promote_to_tensor(tl.sum(tmp128, 0))
    tmp132 = tmp130 + tmp131
    tmp134 = tmp132 + tmp133
    tmp136 = tmp134 + tmp135
    tmp138 = tmp136 * tmp137
    tmp139 = tl.broadcast_to(tmp138, [RBLOCK])
    tmp141 = tl.where(rmask, tmp139, 0)
    tmp142 = triton_helpers.promote_to_tensor(tl.sum(tmp141, 0))
    tmp143 = tl.broadcast_to(tmp136, [RBLOCK])
    tmp145 = tl.where(rmask, tmp143, 0)
    tmp146 = triton_helpers.promote_to_tensor(tl.sum(tmp145, 0))
    tmp149 = tmp147 + tmp148
    tmp151 = tmp149 + tmp150
    tmp153 = tmp151 + tmp152
    tmp155 = tmp153 * tmp154
    tmp156 = tl.broadcast_to(tmp155, [RBLOCK])
    tmp158 = tl.where(rmask, tmp156, 0)
    tmp159 = triton_helpers.promote_to_tensor(tl.sum(tmp158, 0))
    tmp160 = tl.broadcast_to(tmp153, [RBLOCK])
    tmp162 = tl.where(rmask, tmp160, 0)
    tmp163 = triton_helpers.promote_to_tensor(tl.sum(tmp162, 0))
    tmp166 = tmp164 + tmp165
    tmp168 = tmp166 + tmp167
    tmp170 = tmp168 + tmp169
    tmp172 = tmp170 * tmp171
    tmp173 = tl.broadcast_to(tmp172, [RBLOCK])
    tmp175 = tl.where(rmask, tmp173, 0)
    tmp176 = triton_helpers.promote_to_tensor(tl.sum(tmp175, 0))
    tmp177 = tl.broadcast_to(tmp170, [RBLOCK])
    tmp179 = tl.where(rmask, tmp177, 0)
    tmp180 = triton_helpers.promote_to_tensor(tl.sum(tmp179, 0))
    tmp183 = tmp181 + tmp182
    tmp185 = tmp183 + tmp184
    tmp187 = tmp185 + tmp186
    tmp189 = tmp187 * tmp188
    tmp190 = tl.broadcast_to(tmp189, [RBLOCK])
    tmp192 = tl.where(rmask, tmp190, 0)
    tmp193 = triton_helpers.promote_to_tensor(tl.sum(tmp192, 0))
    tmp194 = tl.broadcast_to(tmp187, [RBLOCK])
    tmp196 = tl.where(rmask, tmp194, 0)
    tmp197 = triton_helpers.promote_to_tensor(tl.sum(tmp196, 0))
    tmp198 = tmp6 + tmp23
    tmp199 = tmp198 + tmp40
    tmp200 = tmp199 + tmp57
    tmp201 = tmp200 + tmp74
    tmp202 = tmp201 + tmp91
    tmp203 = tmp202 + tmp125
    tmp204 = tmp203 + tmp159
    tmp205 = tmp204 + tmp193
    tmp206 = tmp205 + tmp108
    tmp207 = tmp206 + tmp142
    tmp208 = tmp207 + tmp176
    tmp209 = tmp10 + tmp27
    tmp210 = tmp209 + tmp44
    tmp211 = tmp210 + tmp61
    tmp212 = tmp211 + tmp78
    tmp213 = tmp212 + tmp95
    tmp214 = tmp213 + tmp129
    tmp215 = tmp214 + tmp163
    tmp216 = tmp215 + tmp197
    tmp217 = tmp216 + tmp112
    tmp218 = tmp217 + tmp146
    tmp219 = tmp218 + tmp180
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp208, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp219, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cw/ccwoinv6oyyujnfebc2waangdkbbdiwx55jsunlmzyobsvnh3o43.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r2) + (524288*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ej/cejrde5ytgb35gyzkwrgq7vrhjxgocunpbu3xemzijvslaayjfzs.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tl.store(in_out_ptr0 + (r1 + (4096*x0)), tmp8, rmask & xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr4 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp12 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = 4096.0
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21 - tmp10
        tmp24 = tmp23 * tmp16
        tmp25 = tmp22 - tmp24
        tmp26 = tmp18 * tmp25
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxwis4qxfaadb6kjjwb634k63rhenzwlmykabnfk6lxb4izku2qc.py
# Source Nodes: [], Original ATen: [aten.add, aten.sum]

triton_per_fused_add_sum_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_sum_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp30 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp35 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp40 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp45 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp50 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp55 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp38 = tl.where(rmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
    tmp43 = tl.where(rmask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None]
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
    tmp48 = tl.where(rmask, tmp46, 0)
    tmp49 = tl.sum(tmp48, 1)[:, None]
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp53 = tl.where(rmask, tmp51, 0)
    tmp54 = tl.sum(tmp53, 1)[:, None]
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, RBLOCK])
    tmp58 = tl.where(rmask, tmp56, 0)
    tmp59 = tl.sum(tmp58, 1)[:, None]
    tmp60 = tmp4 + tmp9
    tmp61 = tmp60 + tmp14
    tmp62 = tmp61 + tmp19
    tmp63 = tmp62 + tmp24
    tmp64 = tmp63 + tmp29
    tmp65 = tmp64 + tmp39
    tmp66 = tmp65 + tmp49
    tmp67 = tmp66 + tmp59
    tmp68 = tmp67 + tmp34
    tmp69 = tmp68 + tmp44
    tmp70 = tmp69 + tmp54
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp70, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdrvpvnga2tmdpe2smenutzwqf32cpv6afggaaqtsqtz7gtijetk.py
# Source Nodes: [], Original ATen: [aten.add, aten.sum]

triton_red_fused_add_sum_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_sum_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp42 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp46 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr5 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr6 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr7 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.load(in_ptr8 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp36 = tl.load(in_ptr9 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp40 = tl.load(in_ptr10 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp44 = tl.load(in_ptr11 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask, tmp39, _tmp38)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(rmask, tmp43, _tmp42)
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
        tmp47 = _tmp46 + tmp45
        _tmp46 = tl.where(rmask, tmp47, _tmp46)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp46 = tl.sum(_tmp46, 1)[:, None]
    tmp48 = tmp2 + tmp6
    tmp49 = tmp48 + tmp10
    tmp50 = tmp49 + tmp14
    tmp51 = tmp50 + tmp18
    tmp52 = tmp51 + tmp22
    tmp53 = tmp52 + tmp30
    tmp54 = tmp53 + tmp38
    tmp55 = tmp54 + tmp46
    tmp56 = tmp55 + tmp26
    tmp57 = tmp56 + tmp34
    tmp58 = tmp57 + tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp58, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/md/cmdbkujov4prz7ip24v7hpwtnlypngoz5owtjgww2dbxumkhvnye.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: 'i32', 39: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(38, 39))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp26 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp29 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp39 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp40 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp42 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp52 = tl.load(in_ptr12 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp53 = tl.load(in_ptr13 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp55 = tl.load(in_ptr14 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp65 = tl.load(in_ptr15 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp66 = tl.load(in_ptr16 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp68 = tl.load(in_ptr17 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp78 = tl.load(in_ptr18 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp79 = tl.load(in_ptr19 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp81 = tl.load(in_ptr20 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp91 = tl.load(in_ptr21 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp92 = tl.load(in_ptr22 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp94 = tl.load(in_ptr23 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp104 = tl.load(in_ptr24 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp105 = tl.load(in_ptr25 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp107 = tl.load(in_ptr26 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp117 = tl.load(in_ptr27 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp118 = tl.load(in_ptr28 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp120 = tl.load(in_ptr29 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp130 = tl.load(in_ptr30 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp131 = tl.load(in_ptr31 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp133 = tl.load(in_ptr32 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp143 = tl.load(in_ptr33 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp144 = tl.load(in_ptr34 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp146 = tl.load(in_ptr35 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp37 = tl.where(rmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp41 = tmp39 + tmp40
    tmp43 = tmp41 * tmp42
    tmp44 = tl.broadcast_to(tmp43, [RBLOCK])
    tmp46 = tl.where(rmask, tmp44, 0)
    tmp47 = triton_helpers.promote_to_tensor(tl.sum(tmp46, 0))
    tmp48 = tl.broadcast_to(tmp41, [RBLOCK])
    tmp50 = tl.where(rmask, tmp48, 0)
    tmp51 = triton_helpers.promote_to_tensor(tl.sum(tmp50, 0))
    tmp54 = tmp52 + tmp53
    tmp56 = tmp54 * tmp55
    tmp57 = tl.broadcast_to(tmp56, [RBLOCK])
    tmp59 = tl.where(rmask, tmp57, 0)
    tmp60 = triton_helpers.promote_to_tensor(tl.sum(tmp59, 0))
    tmp61 = tl.broadcast_to(tmp54, [RBLOCK])
    tmp63 = tl.where(rmask, tmp61, 0)
    tmp64 = triton_helpers.promote_to_tensor(tl.sum(tmp63, 0))
    tmp67 = tmp65 + tmp66
    tmp69 = tmp67 * tmp68
    tmp70 = tl.broadcast_to(tmp69, [RBLOCK])
    tmp72 = tl.where(rmask, tmp70, 0)
    tmp73 = triton_helpers.promote_to_tensor(tl.sum(tmp72, 0))
    tmp74 = tl.broadcast_to(tmp67, [RBLOCK])
    tmp76 = tl.where(rmask, tmp74, 0)
    tmp77 = triton_helpers.promote_to_tensor(tl.sum(tmp76, 0))
    tmp80 = tmp78 + tmp79
    tmp82 = tmp80 * tmp81
    tmp83 = tl.broadcast_to(tmp82, [RBLOCK])
    tmp85 = tl.where(rmask, tmp83, 0)
    tmp86 = triton_helpers.promote_to_tensor(tl.sum(tmp85, 0))
    tmp87 = tl.broadcast_to(tmp80, [RBLOCK])
    tmp89 = tl.where(rmask, tmp87, 0)
    tmp90 = triton_helpers.promote_to_tensor(tl.sum(tmp89, 0))
    tmp93 = tmp91 + tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tl.broadcast_to(tmp95, [RBLOCK])
    tmp98 = tl.where(rmask, tmp96, 0)
    tmp99 = triton_helpers.promote_to_tensor(tl.sum(tmp98, 0))
    tmp100 = tl.broadcast_to(tmp93, [RBLOCK])
    tmp102 = tl.where(rmask, tmp100, 0)
    tmp103 = triton_helpers.promote_to_tensor(tl.sum(tmp102, 0))
    tmp106 = tmp104 + tmp105
    tmp108 = tmp106 * tmp107
    tmp109 = tl.broadcast_to(tmp108, [RBLOCK])
    tmp111 = tl.where(rmask, tmp109, 0)
    tmp112 = triton_helpers.promote_to_tensor(tl.sum(tmp111, 0))
    tmp113 = tl.broadcast_to(tmp106, [RBLOCK])
    tmp115 = tl.where(rmask, tmp113, 0)
    tmp116 = triton_helpers.promote_to_tensor(tl.sum(tmp115, 0))
    tmp119 = tmp117 + tmp118
    tmp121 = tmp119 * tmp120
    tmp122 = tl.broadcast_to(tmp121, [RBLOCK])
    tmp124 = tl.where(rmask, tmp122, 0)
    tmp125 = triton_helpers.promote_to_tensor(tl.sum(tmp124, 0))
    tmp126 = tl.broadcast_to(tmp119, [RBLOCK])
    tmp128 = tl.where(rmask, tmp126, 0)
    tmp129 = triton_helpers.promote_to_tensor(tl.sum(tmp128, 0))
    tmp132 = tmp130 + tmp131
    tmp134 = tmp132 * tmp133
    tmp135 = tl.broadcast_to(tmp134, [RBLOCK])
    tmp137 = tl.where(rmask, tmp135, 0)
    tmp138 = triton_helpers.promote_to_tensor(tl.sum(tmp137, 0))
    tmp139 = tl.broadcast_to(tmp132, [RBLOCK])
    tmp141 = tl.where(rmask, tmp139, 0)
    tmp142 = triton_helpers.promote_to_tensor(tl.sum(tmp141, 0))
    tmp145 = tmp143 + tmp144
    tmp147 = tmp145 * tmp146
    tmp148 = tl.broadcast_to(tmp147, [RBLOCK])
    tmp150 = tl.where(rmask, tmp148, 0)
    tmp151 = triton_helpers.promote_to_tensor(tl.sum(tmp150, 0))
    tmp152 = tl.broadcast_to(tmp145, [RBLOCK])
    tmp154 = tl.where(rmask, tmp152, 0)
    tmp155 = triton_helpers.promote_to_tensor(tl.sum(tmp154, 0))
    tmp156 = tmp8 + tmp21
    tmp157 = tmp156 + tmp34
    tmp158 = tmp157 + tmp47
    tmp159 = tmp158 + tmp60
    tmp160 = tmp159 + tmp73
    tmp161 = tmp160 + tmp99
    tmp162 = tmp161 + tmp125
    tmp163 = tmp162 + tmp151
    tmp164 = tmp163 + tmp86
    tmp165 = tmp164 + tmp112
    tmp166 = tmp165 + tmp138
    tmp167 = tmp12 + tmp25
    tmp168 = tmp167 + tmp38
    tmp169 = tmp168 + tmp51
    tmp170 = tmp169 + tmp64
    tmp171 = tmp170 + tmp77
    tmp172 = tmp171 + tmp103
    tmp173 = tmp172 + tmp129
    tmp174 = tmp173 + tmp155
    tmp175 = tmp174 + tmp90
    tmp176 = tmp175 + tmp116
    tmp177 = tmp176 + tmp142
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp166, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp177, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uh/cuhv342zvsd6qtausk5zapunrdractwueuwkufudkc44zkhfxrex.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r2) + (524288*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ho/chogpdmvc2oxjdn5al7rh3366jv7pmojt6rj5c7pfkztiuozks4f.py
# Source Nodes: [], Original ATen: [aten.add, aten.sum]

triton_per_fused_add_sum_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_sum_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp30 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0)
    tmp35 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0)
    tmp40 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask, other=0.0)
    tmp45 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask, other=0.0)
    tmp50 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask, other=0.0)
    tmp55 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = tl.where(rmask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp41 = tl.broadcast_to(tmp40, [RBLOCK])
    tmp43 = tl.where(rmask, tmp41, 0)
    tmp44 = triton_helpers.promote_to_tensor(tl.sum(tmp43, 0))
    tmp46 = tl.broadcast_to(tmp45, [RBLOCK])
    tmp48 = tl.where(rmask, tmp46, 0)
    tmp49 = triton_helpers.promote_to_tensor(tl.sum(tmp48, 0))
    tmp51 = tl.broadcast_to(tmp50, [RBLOCK])
    tmp53 = tl.where(rmask, tmp51, 0)
    tmp54 = triton_helpers.promote_to_tensor(tl.sum(tmp53, 0))
    tmp56 = tl.broadcast_to(tmp55, [RBLOCK])
    tmp58 = tl.where(rmask, tmp56, 0)
    tmp59 = triton_helpers.promote_to_tensor(tl.sum(tmp58, 0))
    tmp60 = tmp4 + tmp9
    tmp61 = tmp60 + tmp14
    tmp62 = tmp61 + tmp19
    tmp63 = tmp62 + tmp24
    tmp64 = tmp63 + tmp29
    tmp65 = tmp64 + tmp39
    tmp66 = tmp65 + tmp49
    tmp67 = tmp66 + tmp59
    tmp68 = tmp67 + tmp34
    tmp69 = tmp68 + tmp44
    tmp70 = tmp69 + tmp54
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp70, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwv4eurnyebaqsv6hlojivgjcopdv5e36sggwszqxji4rk7qv53.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x0), None)
    tmp11 = tl.load(in_ptr5 + (x0), None)
    tmp13 = tl.load(in_ptr6 + (x0), None)
    tmp15 = tl.load(in_ptr7 + (x0), None)
    tmp17 = tl.load(in_ptr8 + (x0), None)
    tmp19 = tl.load(in_ptr9 + (x0), None)
    tmp21 = tl.load(in_ptr10 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvvl5zm654w5jugc3jjgais2qnbgvol7edxbj4nbuqrzm6ttu75.py
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
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x0), None)
    tmp11 = tl.load(in_ptr5 + (x0), None)
    tmp13 = tl.load(in_ptr6 + (x0), None)
    tmp15 = tl.load(in_ptr7 + (x0), None)
    tmp17 = tl.load(in_ptr8 + (x0), None)
    tmp19 = tl.load(in_ptr9 + (x0), None)
    tmp21 = tl.load(in_ptr10 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bf/cbfn4t34us5wbrbdfkg32ck7ujfrdgthedh6kj7zel7k6b3idgkw.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3exrukgfpev7hnvgoggne324jae4kuaf6d34swkvqrezxakngh.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmbepwae4kic6cgzyxtn2zuavd762kscalkxestkdwdqphulzar.py
# Source Nodes: [start_loss], Original ATen: [aten.embedding_dense_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# start_loss => full_default_2
triton_per_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*i64', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 128.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = tl.full([1, 1], -1, tl.int64)
    tmp22 = tmp20 == tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp22, tmp23, tmp19)
    tmp26 = tmp25 == tmp21
    tmp27 = tl.where(tmp26, tmp23, tmp19)
    tmp29 = tl.full([1, 1], 0, tl.int64)
    tmp30 = tmp28 == tmp29
    tmp31 = tl.where(tmp30, tmp23, tmp19)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp24, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (128*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpb6ba7xhdakytmtkwgxs4dxffuprjkjfyijwd44qeth6pcnkbij.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf3xm67ndsfnuqropefnhi64rt2zaqsejh27r6nflnrakf2wcmnm.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cqug6sjggclgxcgjeioe2trsy6ezjcu2csud2xfq56zpipijawba.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/js/cjsvt5iuy4w2ulj3vivhe2zks6muzvnnh4xwbw52zkl63dutimpa.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_16, primals_22, primals_28, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, sub_39, ne, sub_41, ne_3, ne_6, where_4, ne_8, where_6, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_151, permute_152, alias_29, permute_153, permute_154, permute_159, permute_163, permute_167, div_33, div_34, permute_184, permute_185, alias_31, permute_186, permute_187, div_36, div_37, permute_217, permute_218, alias_33, permute_219, permute_220, div_39, div_40, permute_250, permute_251, alias_35, permute_252, permute_253, div_42, div_43, permute_283, permute_284, alias_37, permute_285, permute_286, div_45, div_46, permute_316, permute_317, alias_39, permute_318, permute_319, div_48, div_49, permute_349, permute_350, alias_41, permute_351, permute_352, div_51, div_52, permute_382, permute_383, alias_43, permute_384, permute_385, div_54, div_55, permute_415, permute_416, alias_45, permute_417, permute_418, div_57, div_58, permute_448, permute_449, alias_47, permute_450, permute_451, div_60, div_61, permute_481, permute_482, alias_49, permute_483, permute_484, div_63, div_64, permute_514, permute_515, alias_51, permute_516, permute_517, permute_534, div_66, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_16, (4096, ), (1, ))
    assert_size_stride(primals_22, (4096, ), (1, ))
    assert_size_stride(primals_28, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (512, 1))
    assert_size_stride(slice_2, (1, 512), (512, 1))
    assert_size_stride(mul_1, (1, 512, 128), (65536, 128, 1))
    assert_size_stride(view, (512, 128), (128, 1))
    assert_size_stride(view_2, (512, 4096), (4096, 1))
    assert_size_stride(view_18, (512, 4096), (4096, 1))
    assert_size_stride(mul_3, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_20, (512, 4096), (4096, 1))
    assert_size_stride(addmm_5, (512, 16384), (16384, 1))
    assert_size_stride(tanh, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_22, (512, 16384), (16384, 1))
    assert_size_stride(mul_9, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_24, (512, 4096), (4096, 1))
    assert_size_stride(view_40, (512, 4096), (4096, 1))
    assert_size_stride(mul_11, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_42, (512, 4096), (4096, 1))
    assert_size_stride(addmm_11, (512, 16384), (16384, 1))
    assert_size_stride(tanh_1, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_44, (512, 16384), (16384, 1))
    assert_size_stride(mul_17, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_46, (512, 4096), (4096, 1))
    assert_size_stride(view_62, (512, 4096), (4096, 1))
    assert_size_stride(mul_19, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_64, (512, 4096), (4096, 1))
    assert_size_stride(addmm_17, (512, 16384), (16384, 1))
    assert_size_stride(tanh_2, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_66, (512, 16384), (16384, 1))
    assert_size_stride(mul_25, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_68, (512, 4096), (4096, 1))
    assert_size_stride(view_84, (512, 4096), (4096, 1))
    assert_size_stride(mul_27, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_86, (512, 4096), (4096, 1))
    assert_size_stride(addmm_23, (512, 16384), (16384, 1))
    assert_size_stride(tanh_3, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_88, (512, 16384), (16384, 1))
    assert_size_stride(mul_33, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_90, (512, 4096), (4096, 1))
    assert_size_stride(view_106, (512, 4096), (4096, 1))
    assert_size_stride(mul_35, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_108, (512, 4096), (4096, 1))
    assert_size_stride(addmm_29, (512, 16384), (16384, 1))
    assert_size_stride(tanh_4, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_110, (512, 16384), (16384, 1))
    assert_size_stride(mul_41, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_112, (512, 4096), (4096, 1))
    assert_size_stride(view_128, (512, 4096), (4096, 1))
    assert_size_stride(mul_43, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_130, (512, 4096), (4096, 1))
    assert_size_stride(addmm_35, (512, 16384), (16384, 1))
    assert_size_stride(tanh_5, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_132, (512, 16384), (16384, 1))
    assert_size_stride(mul_49, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_134, (512, 4096), (4096, 1))
    assert_size_stride(view_150, (512, 4096), (4096, 1))
    assert_size_stride(mul_51, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_152, (512, 4096), (4096, 1))
    assert_size_stride(addmm_41, (512, 16384), (16384, 1))
    assert_size_stride(tanh_6, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_154, (512, 16384), (16384, 1))
    assert_size_stride(mul_57, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_156, (512, 4096), (4096, 1))
    assert_size_stride(view_172, (512, 4096), (4096, 1))
    assert_size_stride(mul_59, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_174, (512, 4096), (4096, 1))
    assert_size_stride(addmm_47, (512, 16384), (16384, 1))
    assert_size_stride(tanh_7, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_176, (512, 16384), (16384, 1))
    assert_size_stride(mul_65, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_178, (512, 4096), (4096, 1))
    assert_size_stride(view_194, (512, 4096), (4096, 1))
    assert_size_stride(mul_67, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_196, (512, 4096), (4096, 1))
    assert_size_stride(addmm_53, (512, 16384), (16384, 1))
    assert_size_stride(tanh_8, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_198, (512, 16384), (16384, 1))
    assert_size_stride(mul_73, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_200, (512, 4096), (4096, 1))
    assert_size_stride(view_216, (512, 4096), (4096, 1))
    assert_size_stride(mul_75, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_218, (512, 4096), (4096, 1))
    assert_size_stride(addmm_59, (512, 16384), (16384, 1))
    assert_size_stride(tanh_9, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_220, (512, 16384), (16384, 1))
    assert_size_stride(mul_81, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_222, (512, 4096), (4096, 1))
    assert_size_stride(view_238, (512, 4096), (4096, 1))
    assert_size_stride(mul_83, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_240, (512, 4096), (4096, 1))
    assert_size_stride(addmm_65, (512, 16384), (16384, 1))
    assert_size_stride(tanh_10, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_242, (512, 16384), (16384, 1))
    assert_size_stride(mul_89, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_244, (512, 4096), (4096, 1))
    assert_size_stride(view_260, (512, 4096), (4096, 1))
    assert_size_stride(mul_91, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_262, (512, 4096), (4096, 1))
    assert_size_stride(addmm_71, (512, 16384), (16384, 1))
    assert_size_stride(tanh_11, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_264, (512, 16384), (16384, 1))
    assert_size_stride(mul_97, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_266, (512, 4096), (4096, 1))
    assert_size_stride(sub_39, (1, 512), (512, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_41, (1, 512), (512, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_4, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_6, (1, 1), (1, 1))
    assert_size_stride(permute_134, (2, 4096), (4096, 1))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_138, (4096, 16384), (16384, 1))
    assert_size_stride(permute_142, (16384, 4096), (4096, 1))
    assert_size_stride(div_31, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_146, (4096, 4096), (4096, 1))
    assert_size_stride(permute_151, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_152, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_29, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_153, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_154, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(permute_159, (4096, 4096), (4096, 1))
    assert_size_stride(permute_163, (4096, 4096), (4096, 1))
    assert_size_stride(permute_167, (4096, 4096), (4096, 1))
    assert_size_stride(div_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_34, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_184, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_185, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_31, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_186, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_187, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_37, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_217, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_218, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_33, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_219, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_220, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_250, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_251, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_35, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_252, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_253, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_283, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_284, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_37, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_285, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_286, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_316, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_317, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_39, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_318, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_319, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_349, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_350, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_41, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_351, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_352, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_382, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_383, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_43, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_384, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_385, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_415, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_416, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_45, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_417, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_418, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_448, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_449, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_47, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_450, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_451, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_481, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_482, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_49, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_483, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_484, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_64, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_514, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_515, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_51, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_516, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_517, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(permute_534, (4096, 128), (128, 1))
    assert_size_stride(div_66, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512), (512, 1))
    assert_size_stride(tangents_3, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_0.run(buf0, 512, grid=grid(512), stream=stream0)
        aten.scatter_(buf0,1,where_4,-1.0)
        del where_4
        buf4 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_0.run(buf4, 512, grid=grid(512), stream=stream0)
        aten.scatter_(buf4,1,where_6,-1.0)
        del where_6
        buf3 = empty((1, 1), device='cuda', dtype=torch.float32)
        buf7 = empty((1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_loss, start_loss], Original ATen: [aten._log_softmax_backward_data, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_1.run(buf0, ne_6, tangents_1, ne_3, buf4, ne_8, ne, buf3, buf7, 1, 512, grid=grid(1), stream=stream0)
        buf8 = empty((1, 512, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_2.run(tangents_2, buf4, ne_8, tangents_1, ne, sub_39, buf7, tangents_3, buf0, ne_6, ne_3, sub_41, buf3, buf8, 1024, grid=grid(1024), stream=stream0)
        del buf0
        del buf3
        del buf4
        del buf7
        del ne
        del ne_3
        del ne_6
        del ne_8
        del sub_39
        del sub_41
        del tangents_1
        del tangents_2
        del tangents_3
        buf9 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (512, 2), (2, 1), 0), permute_134, out=buf9)
        del permute_134
        buf10 = empty((2, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (2, 512), (1, 2), 0), view_266, out=buf10)
        del view_266
        buf11 = empty_strided((1, 2, 4), (8, 1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf8, buf11, 8, 128, grid=grid(8), stream=stream0)
        del buf8
        buf12 = empty((1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf11, buf12, 2, 4, grid=grid(2), stream=stream0)
        del buf11
        buf15 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_5.run(buf9, primals_22, mul_97, div_30, buf15, 512, 4096, grid=grid(512), stream=stream0)
        del div_30
        buf18 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (512, 4096), (4096, 1), 0), permute_138, out=buf18)
        buf22 = reinterpret_tensor(buf18, (1, 512, 16384), (8388608, 16384, 1), 0); del buf18  # reuse
        # Source Nodes: [add_59, mul_45], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf22, addmm_71, tanh_11, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_71
        del tanh_11
        buf23 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (512, 16384), (16384, 1), 0), permute_142, out=buf23)
        buf28 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf15, buf23, primals_16, mul_91, div_31, buf28, 512, 4096, grid=grid(512), stream=stream0)
        del div_31
        buf31 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 4096), (4096, 1), 0), permute_146, out=buf31)
        buf35 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_151, reinterpret_tensor(buf31, (64, 512, 64), (64, 4096, 1), 0), out=buf35)
        del permute_151
        buf41 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf35, buf41, 2097152, grid=grid(2097152), stream=stream0)
        buf42 = reinterpret_tensor(buf35, (512, 4096), (4096, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf41, permute_159, out=buf42)
        buf36 = empty((64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (64, 512, 64), (64, 4096, 1), 0), permute_152, out=buf36)
        del permute_152
        buf38 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf36, alias_29, buf38, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_29
        buf39 = reinterpret_tensor(buf31, (64, 64, 512), (32768, 512, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_153, reinterpret_tensor(buf38, (64, 512, 512), (262144, 512, 1), 0), out=buf39)
        del permute_153
        buf46 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (512, 4096), (1, 512), 0), permute_163, out=buf46)
        buf40 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf38, (64, 512, 512), (262144, 512, 1), 0), permute_154, out=buf40)
        del permute_154
        buf49 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf40, buf49, 2097152, grid=grid(2097152), stream=stream0)
        buf50 = reinterpret_tensor(buf40, (512, 4096), (4096, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf49, permute_167, out=buf50)
        buf54 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf57 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf28, buf42, buf46, buf50, primals_22, mul_89, div_33, buf54, buf57, 512, 4096, grid=grid(512), stream=stream0)
        del div_33
        buf60 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (512, 4096), (4096, 1), 0), permute_138, out=buf60)
        buf64 = reinterpret_tensor(buf60, (1, 512, 16384), (8388608, 16384, 1), 0); del buf60  # reuse
        # Source Nodes: [add_54, mul_41], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf64, addmm_65, tanh_10, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_65
        del tanh_10
        buf65 = reinterpret_tensor(buf54, (512, 4096), (4096, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 16384), (16384, 1), 0), permute_142, out=buf65)
        buf70 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf57, buf65, primals_16, mul_83, div_34, buf70, 512, 4096, grid=grid(512), stream=stream0)
        del div_34
        buf73 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 4096), (4096, 1), 0), permute_146, out=buf73)
        buf77 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_184, reinterpret_tensor(buf73, (64, 512, 64), (64, 4096, 1), 0), out=buf77)
        del permute_184
        buf83 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf77, buf83, 2097152, grid=grid(2097152), stream=stream0)
        buf84 = reinterpret_tensor(buf77, (512, 4096), (4096, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf83, permute_159, out=buf84)
        buf78 = reinterpret_tensor(buf38, (64, 512, 512), (262144, 512, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (64, 512, 64), (64, 4096, 1), 0), permute_185, out=buf78)
        del permute_185
        buf80 = reinterpret_tensor(buf36, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf78, alias_31, buf80, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_31
        buf81 = reinterpret_tensor(buf73, (64, 64, 512), (32768, 512, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_186, reinterpret_tensor(buf80, (64, 512, 512), (262144, 512, 1), 0), out=buf81)
        del permute_186
        buf88 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (512, 4096), (1, 512), 0), permute_163, out=buf88)
        buf82 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf80, (64, 512, 512), (262144, 512, 1), 0), permute_187, out=buf82)
        del permute_187
        buf91 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf82, buf91, 2097152, grid=grid(2097152), stream=stream0)
        buf92 = reinterpret_tensor(buf82, (512, 4096), (4096, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf91, permute_167, out=buf92)
        buf96 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf99 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf70, buf84, buf88, buf92, primals_22, mul_81, div_36, buf96, buf99, 512, 4096, grid=grid(512), stream=stream0)
        del div_36
        buf102 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 4096), (4096, 1), 0), permute_138, out=buf102)
        buf106 = reinterpret_tensor(buf102, (1, 512, 16384), (8388608, 16384, 1), 0); del buf102  # reuse
        # Source Nodes: [add_49, mul_37], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf106, addmm_59, tanh_9, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_59
        del tanh_9
        buf107 = reinterpret_tensor(buf96, (512, 4096), (4096, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (512, 16384), (16384, 1), 0), permute_142, out=buf107)
        buf112 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf99, buf107, primals_16, mul_75, div_37, buf112, 512, 4096, grid=grid(512), stream=stream0)
        del div_37
        buf115 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (512, 4096), (4096, 1), 0), permute_146, out=buf115)
        buf119 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_217, reinterpret_tensor(buf115, (64, 512, 64), (64, 4096, 1), 0), out=buf119)
        del permute_217
        buf125 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf119, buf125, 2097152, grid=grid(2097152), stream=stream0)
        buf126 = reinterpret_tensor(buf119, (512, 4096), (4096, 1), 0); del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf125, permute_159, out=buf126)
        buf120 = reinterpret_tensor(buf80, (64, 512, 512), (262144, 512, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf115, (64, 512, 64), (64, 4096, 1), 0), permute_218, out=buf120)
        del permute_218
        buf122 = reinterpret_tensor(buf78, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf120, alias_33, buf122, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_33
        buf123 = reinterpret_tensor(buf115, (64, 64, 512), (32768, 512, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_219, reinterpret_tensor(buf122, (64, 512, 512), (262144, 512, 1), 0), out=buf123)
        del permute_219
        buf130 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (512, 4096), (1, 512), 0), permute_163, out=buf130)
        buf124 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf122, (64, 512, 512), (262144, 512, 1), 0), permute_220, out=buf124)
        del permute_220
        buf133 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf124, buf133, 2097152, grid=grid(2097152), stream=stream0)
        buf134 = reinterpret_tensor(buf124, (512, 4096), (4096, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf133, permute_167, out=buf134)
        buf138 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf141 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf112, buf126, buf130, buf134, primals_22, mul_73, div_39, buf138, buf141, 512, 4096, grid=grid(512), stream=stream0)
        del div_39
        buf144 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (512, 4096), (4096, 1), 0), permute_138, out=buf144)
        buf148 = reinterpret_tensor(buf144, (1, 512, 16384), (8388608, 16384, 1), 0); del buf144  # reuse
        # Source Nodes: [add_44, mul_33], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf148, addmm_53, tanh_8, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_53
        del tanh_8
        buf149 = reinterpret_tensor(buf138, (512, 4096), (4096, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 16384), (16384, 1), 0), permute_142, out=buf149)
        buf154 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf141, buf149, primals_16, mul_67, div_40, buf154, 512, 4096, grid=grid(512), stream=stream0)
        del div_40
        buf157 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (512, 4096), (4096, 1), 0), permute_146, out=buf157)
        buf161 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_250, reinterpret_tensor(buf157, (64, 512, 64), (64, 4096, 1), 0), out=buf161)
        del permute_250
        buf167 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf161, buf167, 2097152, grid=grid(2097152), stream=stream0)
        buf168 = reinterpret_tensor(buf161, (512, 4096), (4096, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf167, permute_159, out=buf168)
        buf162 = reinterpret_tensor(buf122, (64, 512, 512), (262144, 512, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf157, (64, 512, 64), (64, 4096, 1), 0), permute_251, out=buf162)
        del permute_251
        buf164 = reinterpret_tensor(buf120, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf162, alias_35, buf164, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_35
        buf165 = reinterpret_tensor(buf157, (64, 64, 512), (32768, 512, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_252, reinterpret_tensor(buf164, (64, 512, 512), (262144, 512, 1), 0), out=buf165)
        del permute_252
        buf172 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (512, 4096), (1, 512), 0), permute_163, out=buf172)
        buf166 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf164, (64, 512, 512), (262144, 512, 1), 0), permute_253, out=buf166)
        del permute_253
        buf175 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf166, buf175, 2097152, grid=grid(2097152), stream=stream0)
        buf176 = reinterpret_tensor(buf166, (512, 4096), (4096, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf175, permute_167, out=buf176)
        buf180 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf183 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf154, buf168, buf172, buf176, primals_22, mul_65, div_42, buf180, buf183, 512, 4096, grid=grid(512), stream=stream0)
        del div_42
        buf186 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (512, 4096), (4096, 1), 0), permute_138, out=buf186)
        buf190 = reinterpret_tensor(buf186, (1, 512, 16384), (8388608, 16384, 1), 0); del buf186  # reuse
        # Source Nodes: [add_39, mul_29], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf190, addmm_47, tanh_7, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_47
        del tanh_7
        buf191 = reinterpret_tensor(buf180, (512, 4096), (4096, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (512, 16384), (16384, 1), 0), permute_142, out=buf191)
        buf196 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf183, buf191, primals_16, mul_59, div_43, buf196, 512, 4096, grid=grid(512), stream=stream0)
        del div_43
        buf199 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (512, 4096), (4096, 1), 0), permute_146, out=buf199)
        buf203 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_283, reinterpret_tensor(buf199, (64, 512, 64), (64, 4096, 1), 0), out=buf203)
        del permute_283
        buf209 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf203, buf209, 2097152, grid=grid(2097152), stream=stream0)
        buf210 = reinterpret_tensor(buf203, (512, 4096), (4096, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf209, permute_159, out=buf210)
        buf204 = reinterpret_tensor(buf164, (64, 512, 512), (262144, 512, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf199, (64, 512, 64), (64, 4096, 1), 0), permute_284, out=buf204)
        del permute_284
        buf206 = reinterpret_tensor(buf162, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf204, alias_37, buf206, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_37
        buf207 = reinterpret_tensor(buf199, (64, 64, 512), (32768, 512, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_285, reinterpret_tensor(buf206, (64, 512, 512), (262144, 512, 1), 0), out=buf207)
        del permute_285
        buf214 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (512, 4096), (1, 512), 0), permute_163, out=buf214)
        buf208 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf206, (64, 512, 512), (262144, 512, 1), 0), permute_286, out=buf208)
        del permute_286
        buf217 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf208, buf217, 2097152, grid=grid(2097152), stream=stream0)
        buf218 = reinterpret_tensor(buf208, (512, 4096), (4096, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf217, permute_167, out=buf218)
        buf222 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf225 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf196, buf210, buf214, buf218, primals_22, mul_57, div_45, buf222, buf225, 512, 4096, grid=grid(512), stream=stream0)
        del div_45
        buf228 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (512, 4096), (4096, 1), 0), permute_138, out=buf228)
        buf232 = reinterpret_tensor(buf228, (1, 512, 16384), (8388608, 16384, 1), 0); del buf228  # reuse
        # Source Nodes: [add_34, mul_25], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf232, addmm_41, tanh_6, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_41
        del tanh_6
        buf233 = reinterpret_tensor(buf222, (512, 4096), (4096, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (512, 16384), (16384, 1), 0), permute_142, out=buf233)
        buf238 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf225, buf233, primals_16, mul_51, div_46, buf238, 512, 4096, grid=grid(512), stream=stream0)
        del div_46
        buf241 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (512, 4096), (4096, 1), 0), permute_146, out=buf241)
        buf245 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_316, reinterpret_tensor(buf241, (64, 512, 64), (64, 4096, 1), 0), out=buf245)
        del permute_316
        buf251 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf245, buf251, 2097152, grid=grid(2097152), stream=stream0)
        buf252 = reinterpret_tensor(buf245, (512, 4096), (4096, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf251, permute_159, out=buf252)
        buf246 = reinterpret_tensor(buf206, (64, 512, 512), (262144, 512, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (64, 512, 64), (64, 4096, 1), 0), permute_317, out=buf246)
        del permute_317
        buf248 = reinterpret_tensor(buf204, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf246, alias_39, buf248, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_39
        buf249 = reinterpret_tensor(buf241, (64, 64, 512), (32768, 512, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_318, reinterpret_tensor(buf248, (64, 512, 512), (262144, 512, 1), 0), out=buf249)
        del permute_318
        buf256 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 4096), (1, 512), 0), permute_163, out=buf256)
        buf250 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf248, (64, 512, 512), (262144, 512, 1), 0), permute_319, out=buf250)
        del permute_319
        buf259 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf250, buf259, 2097152, grid=grid(2097152), stream=stream0)
        buf260 = reinterpret_tensor(buf250, (512, 4096), (4096, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf259, permute_167, out=buf260)
        buf264 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf267 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf238, buf252, buf256, buf260, primals_22, mul_49, div_48, buf264, buf267, 512, 4096, grid=grid(512), stream=stream0)
        del div_48
        buf270 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (512, 4096), (4096, 1), 0), permute_138, out=buf270)
        buf274 = reinterpret_tensor(buf270, (1, 512, 16384), (8388608, 16384, 1), 0); del buf270  # reuse
        # Source Nodes: [add_29, mul_21], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf274, addmm_35, tanh_5, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_35
        del tanh_5
        buf275 = reinterpret_tensor(buf264, (512, 4096), (4096, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (512, 16384), (16384, 1), 0), permute_142, out=buf275)
        buf280 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf267, buf275, primals_16, mul_43, div_49, buf280, 512, 4096, grid=grid(512), stream=stream0)
        del div_49
        buf283 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (512, 4096), (4096, 1), 0), permute_146, out=buf283)
        buf287 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_349, reinterpret_tensor(buf283, (64, 512, 64), (64, 4096, 1), 0), out=buf287)
        del permute_349
        buf293 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf287, buf293, 2097152, grid=grid(2097152), stream=stream0)
        buf294 = reinterpret_tensor(buf287, (512, 4096), (4096, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf293, permute_159, out=buf294)
        buf288 = reinterpret_tensor(buf248, (64, 512, 512), (262144, 512, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf283, (64, 512, 64), (64, 4096, 1), 0), permute_350, out=buf288)
        del permute_350
        buf290 = reinterpret_tensor(buf246, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf288, alias_41, buf290, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_41
        buf291 = reinterpret_tensor(buf283, (64, 64, 512), (32768, 512, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_351, reinterpret_tensor(buf290, (64, 512, 512), (262144, 512, 1), 0), out=buf291)
        del permute_351
        buf298 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (512, 4096), (1, 512), 0), permute_163, out=buf298)
        buf292 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf290, (64, 512, 512), (262144, 512, 1), 0), permute_352, out=buf292)
        del permute_352
        buf301 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf292, buf301, 2097152, grid=grid(2097152), stream=stream0)
        buf302 = reinterpret_tensor(buf292, (512, 4096), (4096, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf301, permute_167, out=buf302)
        buf306 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf309 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf280, buf294, buf298, buf302, primals_22, mul_41, div_51, buf306, buf309, 512, 4096, grid=grid(512), stream=stream0)
        del div_51
        buf312 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (512, 4096), (4096, 1), 0), permute_138, out=buf312)
        buf316 = reinterpret_tensor(buf312, (1, 512, 16384), (8388608, 16384, 1), 0); del buf312  # reuse
        # Source Nodes: [add_24, mul_17], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf316, addmm_29, tanh_4, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_29
        del tanh_4
        buf317 = reinterpret_tensor(buf306, (512, 4096), (4096, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (512, 16384), (16384, 1), 0), permute_142, out=buf317)
        buf322 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf309, buf317, primals_16, mul_35, div_52, buf322, 512, 4096, grid=grid(512), stream=stream0)
        del div_52
        buf325 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (512, 4096), (4096, 1), 0), permute_146, out=buf325)
        buf329 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_382, reinterpret_tensor(buf325, (64, 512, 64), (64, 4096, 1), 0), out=buf329)
        del permute_382
        buf335 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf329, buf335, 2097152, grid=grid(2097152), stream=stream0)
        buf336 = reinterpret_tensor(buf329, (512, 4096), (4096, 1), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf335, permute_159, out=buf336)
        buf330 = reinterpret_tensor(buf290, (64, 512, 512), (262144, 512, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (64, 512, 64), (64, 4096, 1), 0), permute_383, out=buf330)
        del permute_383
        buf332 = reinterpret_tensor(buf288, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf330, alias_43, buf332, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_43
        buf333 = reinterpret_tensor(buf325, (64, 64, 512), (32768, 512, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_384, reinterpret_tensor(buf332, (64, 512, 512), (262144, 512, 1), 0), out=buf333)
        del permute_384
        buf340 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (512, 4096), (1, 512), 0), permute_163, out=buf340)
        buf334 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf332, (64, 512, 512), (262144, 512, 1), 0), permute_385, out=buf334)
        del permute_385
        buf343 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf334, buf343, 2097152, grid=grid(2097152), stream=stream0)
        buf344 = reinterpret_tensor(buf334, (512, 4096), (4096, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf343, permute_167, out=buf344)
        buf348 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf351 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf322, buf336, buf340, buf344, primals_22, mul_33, div_54, buf348, buf351, 512, 4096, grid=grid(512), stream=stream0)
        del div_54
        buf356 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf351, (512, 4096), (4096, 1), 0), permute_138, out=buf356)
        buf362 = reinterpret_tensor(buf356, (1, 512, 16384), (8388608, 16384, 1), 0); del buf356  # reuse
        # Source Nodes: [add_19, mul_13], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf362, addmm_23, tanh_3, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_23
        del tanh_3
        buf363 = reinterpret_tensor(buf348, (512, 4096), (4096, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (512, 16384), (16384, 1), 0), permute_142, out=buf363)
        buf370 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf351, buf363, primals_16, mul_27, div_55, buf370, 512, 4096, grid=grid(512), stream=stream0)
        del div_55
        buf375 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (512, 4096), (4096, 1), 0), permute_146, out=buf375)
        buf381 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_415, reinterpret_tensor(buf375, (64, 512, 64), (64, 4096, 1), 0), out=buf381)
        del permute_415
        buf387 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf381, buf387, 2097152, grid=grid(2097152), stream=stream0)
        buf388 = reinterpret_tensor(buf381, (512, 4096), (4096, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf387, permute_159, out=buf388)
        buf382 = reinterpret_tensor(buf332, (64, 512, 512), (262144, 512, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf375, (64, 512, 64), (64, 4096, 1), 0), permute_416, out=buf382)
        del permute_416
        buf384 = reinterpret_tensor(buf330, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf382, alias_45, buf384, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_45
        buf385 = reinterpret_tensor(buf375, (64, 64, 512), (32768, 512, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_417, reinterpret_tensor(buf384, (64, 512, 512), (262144, 512, 1), 0), out=buf385)
        del permute_417
        buf394 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf385, (512, 4096), (1, 512), 0), permute_163, out=buf394)
        buf386 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf384, (64, 512, 512), (262144, 512, 1), 0), permute_418, out=buf386)
        del permute_418
        buf399 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf386, buf399, 2097152, grid=grid(2097152), stream=stream0)
        buf400 = reinterpret_tensor(buf386, (512, 4096), (4096, 1), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf399, permute_167, out=buf400)
        buf406 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf409 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf370, buf388, buf394, buf400, primals_22, mul_25, div_57, buf406, buf409, 512, 4096, grid=grid(512), stream=stream0)
        del div_57
        buf412 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (512, 4096), (4096, 1), 0), permute_138, out=buf412)
        buf416 = reinterpret_tensor(buf412, (1, 512, 16384), (8388608, 16384, 1), 0); del buf412  # reuse
        # Source Nodes: [add_14, mul_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf416, addmm_17, tanh_2, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_17
        del tanh_2
        buf417 = reinterpret_tensor(buf406, (512, 4096), (4096, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (512, 16384), (16384, 1), 0), permute_142, out=buf417)
        buf422 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf409, buf417, primals_16, mul_19, div_58, buf422, 512, 4096, grid=grid(512), stream=stream0)
        del div_58
        buf425 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (512, 4096), (4096, 1), 0), permute_146, out=buf425)
        buf429 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_448, reinterpret_tensor(buf425, (64, 512, 64), (64, 4096, 1), 0), out=buf429)
        del permute_448
        buf435 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf429, buf435, 2097152, grid=grid(2097152), stream=stream0)
        buf436 = reinterpret_tensor(buf429, (512, 4096), (4096, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf435, permute_159, out=buf436)
        buf430 = reinterpret_tensor(buf384, (64, 512, 512), (262144, 512, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf425, (64, 512, 64), (64, 4096, 1), 0), permute_449, out=buf430)
        del permute_449
        buf432 = reinterpret_tensor(buf382, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf430, alias_47, buf432, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_47
        buf433 = reinterpret_tensor(buf425, (64, 64, 512), (32768, 512, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_450, reinterpret_tensor(buf432, (64, 512, 512), (262144, 512, 1), 0), out=buf433)
        del permute_450
        buf440 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (512, 4096), (1, 512), 0), permute_163, out=buf440)
        buf434 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf432, (64, 512, 512), (262144, 512, 1), 0), permute_451, out=buf434)
        del permute_451
        buf443 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf434, buf443, 2097152, grid=grid(2097152), stream=stream0)
        buf444 = reinterpret_tensor(buf434, (512, 4096), (4096, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf443, permute_167, out=buf444)
        buf448 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf451 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf422, buf436, buf440, buf444, primals_22, mul_17, div_60, buf448, buf451, 512, 4096, grid=grid(512), stream=stream0)
        del div_60
        buf454 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf451, (512, 4096), (4096, 1), 0), permute_138, out=buf454)
        buf458 = reinterpret_tensor(buf454, (1, 512, 16384), (8388608, 16384, 1), 0); del buf454  # reuse
        # Source Nodes: [add_9, mul_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf458, addmm_11, tanh_1, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_11
        del tanh_1
        buf459 = reinterpret_tensor(buf448, (512, 4096), (4096, 1), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf458, (512, 16384), (16384, 1), 0), permute_142, out=buf459)
        buf464 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf451, buf459, primals_16, mul_11, div_61, buf464, 512, 4096, grid=grid(512), stream=stream0)
        del div_61
        buf467 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (512, 4096), (4096, 1), 0), permute_146, out=buf467)
        buf471 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_481, reinterpret_tensor(buf467, (64, 512, 64), (64, 4096, 1), 0), out=buf471)
        del permute_481
        buf477 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf471, buf477, 2097152, grid=grid(2097152), stream=stream0)
        buf478 = reinterpret_tensor(buf471, (512, 4096), (4096, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf477, permute_159, out=buf478)
        buf472 = reinterpret_tensor(buf432, (64, 512, 512), (262144, 512, 1), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf467, (64, 512, 64), (64, 4096, 1), 0), permute_482, out=buf472)
        del permute_482
        buf474 = reinterpret_tensor(buf430, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf472, alias_49, buf474, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_49
        buf475 = reinterpret_tensor(buf467, (64, 64, 512), (32768, 512, 1), 0); del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_483, reinterpret_tensor(buf474, (64, 512, 512), (262144, 512, 1), 0), out=buf475)
        del permute_483
        buf482 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (512, 4096), (1, 512), 0), permute_163, out=buf482)
        buf476 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf474, (64, 512, 512), (262144, 512, 1), 0), permute_484, out=buf476)
        del permute_484
        buf485 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf476, buf485, 2097152, grid=grid(2097152), stream=stream0)
        buf486 = reinterpret_tensor(buf476, (512, 4096), (4096, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf485, permute_167, out=buf486)
        buf100 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf101 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf354 = buf100; del buf100  # reuse
        buf496 = buf354; del buf354  # reuse
        buf355 = buf101; del buf101  # reuse
        buf497 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_11.run(buf496, buf497, buf9, mul_97, buf28, buf42, buf46, buf50, mul_89, buf70, buf84, buf88, buf92, mul_81, buf112, buf126, buf130, buf134, mul_73, buf154, buf168, buf172, buf176, mul_65, buf196, buf210, buf214, buf218, mul_57, buf370, buf388, buf394, buf400, mul_25, buf238, buf252, buf256, buf260, mul_49, buf422, buf436, buf440, buf444, mul_17, buf280, buf294, buf298, buf302, mul_41, buf464, buf478, buf482, buf486, mul_9, buf322, buf336, buf340, buf344, mul_33, 4096, 512, grid=grid(4096), stream=stream0)
        del buf126
        del buf130
        del buf134
        del buf168
        del buf172
        del buf176
        del buf210
        del buf214
        del buf218
        del buf252
        del buf256
        del buf260
        del buf294
        del buf298
        del buf302
        del buf336
        del buf340
        del buf344
        del buf388
        del buf394
        del buf400
        del buf42
        del buf436
        del buf440
        del buf444
        del buf46
        del buf50
        del buf84
        del buf88
        del buf9
        del mul_17
        del mul_25
        del mul_33
        del mul_41
        del mul_49
        del mul_57
        del mul_65
        del mul_73
        del mul_81
        del mul_89
        del mul_97
        buf19 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (4096, 512), (1, 4096), 0), view_264, out=buf19)
        del view_264
        buf20 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf15, buf20, 16384, 128, grid=grid(16384), stream=stream0)
        buf104 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf99, buf104, 16384, 128, grid=grid(16384), stream=stream0)
        buf146 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf141, buf146, 16384, 128, grid=grid(16384), stream=stream0)
        buf188 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf183, buf188, 16384, 128, grid=grid(16384), stream=stream0)
        buf230 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf225, buf230, 16384, 128, grid=grid(16384), stream=stream0)
        buf272 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf267, buf272, 16384, 128, grid=grid(16384), stream=stream0)
        buf314 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf309, buf314, 16384, 128, grid=grid(16384), stream=stream0)
        buf358 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf351, buf358, 16384, 128, grid=grid(16384), stream=stream0)
        buf414 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf409, buf414, 16384, 128, grid=grid(16384), stream=stream0)
        buf456 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf451, buf456, 16384, 128, grid=grid(16384), stream=stream0)
        buf490 = reinterpret_tensor(buf478, (1, 512, 4096), (2097152, 4096, 1), 0); del buf478  # reuse
        buf493 = reinterpret_tensor(buf92, (1, 512, 4096), (2097152, 4096, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf490, buf464, buf482, buf486, primals_22, mul_9, div_63, buf493, 512, 4096, grid=grid(512), stream=stream0)
        del buf482
        del buf486
        del div_63
        del mul_9
        del primals_22
        buf500 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf493, buf500, 16384, 128, grid=grid(16384), stream=stream0)
        buf62 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf57, buf62, 16384, 128, grid=grid(16384), stream=stream0)
        buf105 = empty((1, 4096), device='cuda', dtype=torch.float32)
        buf360 = reinterpret_tensor(buf105, (4096, ), (1, ), 0); del buf105  # reuse
        buf502 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_14.run(buf502, buf20, buf62, buf104, buf146, buf188, buf230, buf414, buf272, buf456, buf314, buf500, buf358, 4096, 4, grid=grid(4096), stream=stream0)
        buf24 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (16384, 512), (1, 16384), 0), view_262, out=buf24)
        del view_262
        buf498 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf493, (512, 4096), (4096, 1), 0), permute_138, out=buf498)
        del permute_138
        buf504 = reinterpret_tensor(buf498, (1, 512, 16384), (8388608, 16384, 1), 0); del buf498  # reuse
        # Source Nodes: [add_4, mul_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_6.run(buf504, addmm_5, tanh, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_5
        del tanh
        buf109 = reinterpret_tensor(buf62, (1, 16384), (16384, 1), 0); del buf62  # reuse
        buf366 = reinterpret_tensor(buf109, (16384, ), (1, ), 0); del buf109  # reuse
        buf508 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_red_fused_add_sum_15.run(buf508, buf22, buf64, buf106, buf148, buf190, buf232, buf416, buf274, buf458, buf316, buf504, buf362, 16384, 512, grid=grid(16384), stream=stream0)
        del buf22
        buf505 = reinterpret_tensor(buf490, (512, 4096), (4096, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (512, 16384), (16384, 1), 0), permute_142, out=buf505)
        del permute_142
        buf113 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf114 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf373 = buf113; del buf113  # reuse
        buf515 = buf373; del buf373  # reuse
        buf374 = buf114; del buf114  # reuse
        buf516 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf515, buf516, buf15, buf23, mul_91, buf57, buf65, mul_83, buf99, buf107, mul_75, buf141, buf149, mul_67, buf183, buf191, mul_59, buf225, buf233, mul_51, buf409, buf417, mul_19, buf267, buf275, mul_43, buf451, buf459, mul_11, buf309, buf317, mul_35, buf493, buf505, mul_3, buf351, buf363, mul_27, 4096, 512, grid=grid(4096), stream=stream0)
        del buf107
        del buf149
        del buf15
        del buf191
        del buf23
        del buf233
        del buf275
        del buf317
        del buf363
        del buf417
        del buf459
        del mul_11
        del mul_19
        del mul_27
        del mul_35
        del mul_43
        del mul_51
        del mul_59
        del mul_67
        del mul_75
        del mul_83
        del mul_91
        buf32 = reinterpret_tensor(buf474, (4096, 4096), (4096, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (4096, 512), (1, 4096), 0), view_260, out=buf32)
        del view_260
        buf33 = buf500; del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17.run(buf28, buf33, 16384, 128, grid=grid(16384), stream=stream0)
        buf117 = buf456; del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf112, buf117, 16384, 128, grid=grid(16384), stream=stream0)
        buf159 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf154, buf159, 16384, 128, grid=grid(16384), stream=stream0)
        buf201 = buf358; del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf196, buf201, 16384, 128, grid=grid(16384), stream=stream0)
        buf243 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf238, buf243, 16384, 128, grid=grid(16384), stream=stream0)
        buf285 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf280, buf285, 16384, 128, grid=grid(16384), stream=stream0)
        buf327 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf322, buf327, 16384, 128, grid=grid(16384), stream=stream0)
        buf377 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf370, buf377, 16384, 128, grid=grid(16384), stream=stream0)
        buf427 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf422, buf427, 16384, 128, grid=grid(16384), stream=stream0)
        buf469 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf464, buf469, 16384, 128, grid=grid(16384), stream=stream0)
        buf512 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_7.run(buf493, buf505, primals_16, mul_3, div_64, buf512, 512, 4096, grid=grid(512), stream=stream0)
        del div_64
        del mul_3
        del primals_16
        buf519 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf512, buf519, 16384, 128, grid=grid(16384), stream=stream0)
        buf75 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf70, buf75, 16384, 128, grid=grid(16384), stream=stream0)
        buf118 = empty((1, 4096), device='cuda', dtype=torch.float32)
        buf379 = reinterpret_tensor(buf118, (4096, ), (1, ), 0); del buf118  # reuse
        buf521 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_14.run(buf521, buf33, buf75, buf117, buf159, buf201, buf243, buf427, buf285, buf469, buf327, buf519, buf377, 4096, 4, grid=grid(4096), stream=stream0)
        buf43 = reinterpret_tensor(buf472, (4096, 4096), (4096, 1), 0); del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (4096, 512), (1, 4096), 0), view_244, out=buf43)
        buf44 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17.run(buf41, buf44, 16384, 128, grid=grid(16384), stream=stream0)
        buf128 = buf519; del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf125, buf128, 16384, 128, grid=grid(16384), stream=stream0)
        buf170 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf167, buf170, 16384, 128, grid=grid(16384), stream=stream0)
        buf212 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf209, buf212, 16384, 128, grid=grid(16384), stream=stream0)
        buf254 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf251, buf254, 16384, 128, grid=grid(16384), stream=stream0)
        buf296 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf293, buf296, 16384, 128, grid=grid(16384), stream=stream0)
        buf338 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf335, buf338, 16384, 128, grid=grid(16384), stream=stream0)
        buf390 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf387, buf390, 16384, 128, grid=grid(16384), stream=stream0)
        buf438 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf435, buf438, 16384, 128, grid=grid(16384), stream=stream0)
        buf480 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf477, buf480, 16384, 128, grid=grid(16384), stream=stream0)
        buf517 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf512, (512, 4096), (4096, 1), 0), permute_146, out=buf517)
        del permute_146
        buf523 = reinterpret_tensor(buf505, (64, 512, 64), (32768, 64, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_514, reinterpret_tensor(buf517, (64, 512, 64), (64, 4096, 1), 0), out=buf523)
        del permute_514
        buf529 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf523, buf529, 2097152, grid=grid(2097152), stream=stream0)
        del buf523
        buf532 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf529, buf532, 16384, 128, grid=grid(16384), stream=stream0)
        buf86 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf83, buf86, 16384, 128, grid=grid(16384), stream=stream0)
        buf129 = empty((1, 4096), device='cuda', dtype=torch.float32)
        buf392 = reinterpret_tensor(buf129, (4096, ), (1, ), 0); del buf129  # reuse
        buf534 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_14.run(buf534, buf44, buf86, buf128, buf170, buf212, buf254, buf438, buf296, buf480, buf338, buf532, buf390, 4096, 4, grid=grid(4096), stream=stream0)
        buf47 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (4096, 512), (512, 1), 0), view_244, out=buf47)
        buf524 = empty((64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf517, (64, 512, 64), (64, 4096, 1), 0), permute_515, out=buf524)
        del permute_515
        buf526 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_9.run(buf524, alias_51, buf526, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_51
        buf527 = reinterpret_tensor(buf517, (64, 64, 512), (32768, 512, 1), 0); del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_516, reinterpret_tensor(buf526, (64, 512, 512), (262144, 512, 1), 0), out=buf527)
        del permute_516
        buf132 = empty((1, 4096), device='cuda', dtype=torch.float32)
        buf397 = reinterpret_tensor(buf132, (4096, ), (1, ), 0); del buf132  # reuse
        buf539 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_18.run(buf539, buf39, buf81, buf123, buf165, buf207, buf249, buf433, buf291, buf475, buf333, buf527, buf385, 4096, 512, grid=grid(4096), stream=stream0)
        buf51 = reinterpret_tensor(buf524, (4096, 4096), (4096, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (4096, 512), (1, 4096), 0), view_244, out=buf51)
        del view_244
        buf52 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17.run(buf49, buf52, 16384, 128, grid=grid(16384), stream=stream0)
        buf136 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf133, buf136, 16384, 128, grid=grid(16384), stream=stream0)
        buf178 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf175, buf178, 16384, 128, grid=grid(16384), stream=stream0)
        buf220 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf217, buf220, 16384, 128, grid=grid(16384), stream=stream0)
        buf262 = buf438; del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf259, buf262, 16384, 128, grid=grid(16384), stream=stream0)
        buf304 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf301, buf304, 16384, 128, grid=grid(16384), stream=stream0)
        buf346 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf343, buf346, 16384, 128, grid=grid(16384), stream=stream0)
        buf402 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf399, buf402, 16384, 128, grid=grid(16384), stream=stream0)
        buf446 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf443, buf446, 16384, 128, grid=grid(16384), stream=stream0)
        buf488 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf485, buf488, 16384, 128, grid=grid(16384), stream=stream0)
        buf528 = reinterpret_tensor(buf49, (64, 512, 64), (32768, 64, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf526, (64, 512, 512), (262144, 512, 1), 0), permute_517, out=buf528)
        del permute_517
        buf541 = reinterpret_tensor(buf39, (512, 4096), (4096, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf528, buf541, 2097152, grid=grid(2097152), stream=stream0)
        del buf528
        buf544 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf541, buf544, 16384, 128, grid=grid(16384), stream=stream0)
        buf94 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf91, buf94, 16384, 128, grid=grid(16384), stream=stream0)
        buf137 = empty((1, 4096), device='cuda', dtype=torch.float32)
        buf404 = reinterpret_tensor(buf137, (4096, ), (1, ), 0); del buf137  # reuse
        buf546 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_14.run(buf546, buf52, buf94, buf136, buf178, buf220, buf262, buf446, buf304, buf488, buf346, buf544, buf402, 4096, 4, grid=grid(4096), stream=stream0)
        del buf136
        del buf178
        del buf220
        del buf262
        del buf304
        del buf346
        del buf402
        del buf446
        del buf488
        del buf52
        del buf544
        buf61 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (4096, 512), (1, 4096), 0), view_242, out=buf61)
        del buf57
        del view_242
        buf66 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (16384, 512), (1, 16384), 0), view_240, out=buf66)
        del buf64
        del view_240
        buf74 = reinterpret_tensor(buf526, (4096, 4096), (4096, 1), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (4096, 512), (1, 4096), 0), view_238, out=buf74)
        del buf70
        del view_238
        buf85 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (4096, 512), (1, 4096), 0), view_222, out=buf85)
        del buf83
        buf89 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (4096, 512), (512, 1), 0), view_222, out=buf89)
        del buf81
        buf93 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (4096, 512), (1, 4096), 0), view_222, out=buf93)
        del buf91
        del view_222
        buf103 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (4096, 512), (1, 4096), 0), view_220, out=buf103)
        del buf99
        del view_220
        buf108 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (16384, 512), (1, 16384), 0), view_218, out=buf108)
        del buf106
        del view_218
        buf116 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (4096, 512), (1, 4096), 0), view_216, out=buf116)
        del buf112
        del view_216
        buf127 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (4096, 512), (1, 4096), 0), view_200, out=buf127)
        del buf125
        buf131 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (4096, 512), (512, 1), 0), view_200, out=buf131)
        del buf123
        buf135 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (4096, 512), (1, 4096), 0), view_200, out=buf135)
        del buf133
        del view_200
        buf145 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (4096, 512), (1, 4096), 0), view_198, out=buf145)
        del buf141
        del view_198
        buf150 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (16384, 512), (1, 16384), 0), view_196, out=buf150)
        del buf148
        del view_196
        buf158 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (4096, 512), (1, 4096), 0), view_194, out=buf158)
        del buf154
        del view_194
        buf169 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (4096, 512), (1, 4096), 0), view_178, out=buf169)
        del buf167
        buf173 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (4096, 512), (512, 1), 0), view_178, out=buf173)
        del buf165
        buf177 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (4096, 512), (1, 4096), 0), view_178, out=buf177)
        del buf175
        del view_178
        buf187 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (4096, 512), (1, 4096), 0), view_176, out=buf187)
        del buf183
        del view_176
        buf192 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (16384, 512), (1, 16384), 0), view_174, out=buf192)
        del buf190
        del view_174
        buf200 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (4096, 512), (1, 4096), 0), view_172, out=buf200)
        del buf196
        del view_172
        buf211 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (4096, 512), (1, 4096), 0), view_156, out=buf211)
        del buf209
        buf215 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (4096, 512), (512, 1), 0), view_156, out=buf215)
        del buf207
        buf219 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (4096, 512), (1, 4096), 0), view_156, out=buf219)
        del buf217
        del view_156
        buf229 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (4096, 512), (1, 4096), 0), view_154, out=buf229)
        del buf225
        del view_154
        buf234 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (16384, 512), (1, 16384), 0), view_152, out=buf234)
        del buf232
        del view_152
        buf242 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (4096, 512), (1, 4096), 0), view_150, out=buf242)
        del buf238
        del view_150
        buf253 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (4096, 512), (1, 4096), 0), view_134, out=buf253)
        del buf251
        buf257 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (4096, 512), (512, 1), 0), view_134, out=buf257)
        del buf249
        buf261 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (4096, 512), (1, 4096), 0), view_134, out=buf261)
        del buf259
        del view_134
        buf271 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (4096, 512), (1, 4096), 0), view_132, out=buf271)
        del buf267
        del view_132
        buf276 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (16384, 512), (1, 16384), 0), view_130, out=buf276)
        del buf274
        del view_130
        buf284 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (4096, 512), (1, 4096), 0), view_128, out=buf284)
        del buf280
        del view_128
        buf295 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (4096, 512), (1, 4096), 0), view_112, out=buf295)
        del buf293
        buf299 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (4096, 512), (512, 1), 0), view_112, out=buf299)
        del buf291
        buf303 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (4096, 512), (1, 4096), 0), view_112, out=buf303)
        del buf301
        del view_112
        buf313 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (4096, 512), (1, 4096), 0), view_110, out=buf313)
        del buf309
        del view_110
        buf318 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (16384, 512), (1, 16384), 0), view_108, out=buf318)
        del buf316
        del view_108
        buf326 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (4096, 512), (1, 4096), 0), view_106, out=buf326)
        del buf322
        del view_106
        buf337 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (4096, 512), (1, 4096), 0), view_90, out=buf337)
        del buf335
        buf341 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (4096, 512), (512, 1), 0), view_90, out=buf341)
        del buf333
        buf345 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf343, (4096, 512), (1, 4096), 0), view_90, out=buf345)
        del buf343
        del view_90
        buf357 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf351, (4096, 512), (1, 4096), 0), view_88, out=buf357)
        del buf351
        del view_88
        buf413 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (4096, 512), (1, 4096), 0), view_66, out=buf413)
        del buf409
        del view_66
        buf455 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf451, (4096, 512), (1, 4096), 0), view_44, out=buf455)
        del buf451
        del view_44
        buf499 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf493, (4096, 512), (1, 4096), 0), view_22, out=buf499)
        del buf493
        del view_22
        buf361 = buf103; del buf103  # reuse
        buf503 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf503, buf19, buf61, buf145, buf187, buf229, buf271, buf313, buf357, buf413, buf455, buf499, 67108864, grid=grid(67108864), stream=stream0)
        del buf145
        del buf187
        del buf19
        del buf229
        del buf271
        del buf313
        del buf357
        buf364 = reinterpret_tensor(buf61, (16384, 4096), (4096, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (16384, 512), (1, 16384), 0), view_86, out=buf364)
        del buf362
        del view_86
        buf418 = reinterpret_tensor(buf499, (16384, 4096), (4096, 1), 0); del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (16384, 512), (1, 16384), 0), view_64, out=buf418)
        del buf416
        del view_64
        buf460 = reinterpret_tensor(buf455, (16384, 4096), (4096, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf458, (16384, 512), (1, 16384), 0), view_42, out=buf460)
        del buf458
        del view_42
        buf506 = reinterpret_tensor(buf413, (16384, 4096), (4096, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (16384, 512), (1, 16384), 0), view_20, out=buf506)
        del buf504
        del view_20
        buf367 = buf108; del buf108  # reuse
        buf509 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf509, buf24, buf66, buf150, buf192, buf234, buf276, buf318, buf364, buf418, buf460, buf506, 67108864, grid=grid(67108864), stream=stream0)
        del buf150
        del buf192
        del buf234
        del buf24
        del buf276
        del buf318
        del buf364
        del buf418
        del buf460
        del buf506
        del buf66
        buf376 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (4096, 512), (1, 4096), 0), view_84, out=buf376)
        del buf370
        del view_84
        buf426 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (4096, 512), (1, 4096), 0), view_62, out=buf426)
        del buf422
        del view_62
        buf468 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (4096, 512), (1, 4096), 0), view_40, out=buf468)
        del buf464
        del view_40
        buf518 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf512, (4096, 512), (1, 4096), 0), view_18, out=buf518)
        del view_18
        buf380 = buf116; del buf116  # reuse
        buf522 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_20.run(buf522, buf32, buf74, buf158, buf200, buf242, buf284, buf326, buf376, buf426, buf468, buf518, 16777216, grid=grid(16777216), stream=stream0)
        del buf158
        del buf200
        del buf242
        del buf284
        del buf32
        del buf326
        del buf376
        buf389 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (4096, 512), (1, 4096), 0), view_68, out=buf389)
        del buf387
        buf437 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (4096, 512), (1, 4096), 0), view_46, out=buf437)
        del buf435
        buf479 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf477, (4096, 512), (1, 4096), 0), view_24, out=buf479)
        del buf477
        buf531 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf529, (4096, 512), (1, 4096), 0), view_2, out=buf531)
        buf393 = buf127; del buf127  # reuse
        buf535 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_20.run(buf535, buf43, buf85, buf169, buf211, buf253, buf295, buf337, buf389, buf437, buf479, buf531, 16777216, grid=grid(16777216), stream=stream0)
        del buf169
        del buf211
        del buf253
        del buf295
        del buf337
        del buf389
        del buf43
        buf395 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf385, (4096, 512), (512, 1), 0), view_68, out=buf395)
        del buf385
        buf441 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (4096, 512), (512, 1), 0), view_46, out=buf441)
        del buf433
        buf483 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (4096, 512), (512, 1), 0), view_24, out=buf483)
        del buf475
        buf537 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (4096, 512), (512, 1), 0), view_2, out=buf537)
        buf398 = buf131; del buf131  # reuse
        buf540 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_20.run(buf540, buf47, buf89, buf173, buf215, buf257, buf299, buf341, buf395, buf441, buf483, buf537, 16777216, grid=grid(16777216), stream=stream0)
        del buf173
        del buf215
        del buf257
        del buf299
        del buf341
        del buf395
        del buf441
        buf401 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (4096, 512), (1, 4096), 0), view_68, out=buf401)
        del buf399
        del view_68
        buf445 = buf537; del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (4096, 512), (1, 4096), 0), view_46, out=buf445)
        del buf443
        del view_46
        buf487 = buf483; del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (4096, 512), (1, 4096), 0), view_24, out=buf487)
        del view_24
        buf543 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf541, (4096, 512), (1, 4096), 0), view_2, out=buf543)
        del view_2
        buf405 = buf135; del buf135  # reuse
        buf547 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_20.run(buf547, buf51, buf93, buf177, buf219, buf261, buf303, buf345, buf401, buf445, buf487, buf543, 16777216, grid=grid(16777216), stream=stream0)
        del buf177
        del buf219
        del buf261
        del buf303
        del buf345
        del buf401
        del buf445
        del buf487
        del buf51
        del buf543
        del buf93
        buf530 = buf485; del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf529, permute_159, out=buf530)
        del permute_159
        buf536 = buf529; del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (512, 4096), (1, 512), 0), permute_163, out=buf536)
        del permute_163
        buf542 = reinterpret_tensor(buf527, (512, 4096), (4096, 1), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf541, permute_167, out=buf542)
        del buf541
        del permute_167
        buf548 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf548, buf530, buf536, buf542, 2097152, grid=grid(2097152), stream=stream0)
        del buf530
        del buf536
        del buf542
        buf549 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf548, (512, 4096), (4096, 1), 0), permute_534, out=buf549)
        del permute_534
        buf550 = empty((4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf548, (4096, 512), (1, 4096), 0), view, out=buf550)
        del view
        buf551 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17.run(buf548, buf551, 16384, 128, grid=grid(16384), stream=stream0)
        del buf548
        buf552 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_22.run(buf551, buf552, 4096, 4, grid=grid(4096), stream=stream0)
        del buf551
        buf559 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf563 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf567 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_loss], Original ATen: [aten.embedding_dense_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_23.run(buf549, primals_4, mul_1, div_66, slice_2, expand, primals_28, buf559, buf563, buf567, 512, 128, grid=grid(512), stream=stream0)
        del div_66
        del primals_4
        buf556 = empty((128, ), device='cuda', dtype=torch.float32)
        buf557 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_24.run(buf549, mul_1, buf556, buf557, 128, 512, grid=grid(128), stream=stream0)
        del mul_1
        buf558 = buf549; del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_25.run(buf558, 65536, grid=grid(65536), stream=stream0)
        aten.index_put_(buf558, [slice_2], buf559, True)
        del buf559
        del slice_2
        buf562 = empty((2, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_26.run(buf562, 256, grid=grid(256), stream=stream0)
        aten.index_put_(buf562, [expand], buf563, True)
        del buf563
        del expand
        buf566 = empty((30000, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_27.run(buf566, 3840000, grid=grid(3840000), stream=stream0)
        aten.index_put_(buf566, [primals_28], buf567, True)
        del buf567
        del primals_28
        return (buf566, buf562, buf558, buf556, buf557, reinterpret_tensor(buf550, (4096, 128), (128, 1), 0), reinterpret_tensor(buf552, (4096, ), (1, ), 0), buf547, buf546, buf540, buf539, buf535, buf534, buf522, buf521, buf515, buf516, buf509, buf508, buf503, buf502, buf496, buf497, reinterpret_tensor(buf10, (2, 4096), (4096, 1), 0), reinterpret_tensor(buf12, (2, ), (1, ), 0), None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    expand = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    slice_2 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    mul_1 = rand_strided((1, 512, 128), (65536, 128, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_3 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_24 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_11 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_11 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_1 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_17 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_46 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_19 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_2 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_25 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_27 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_3 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_33 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_29 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_4 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_41 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_112 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_43 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_35 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_5 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_49 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_51 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_41 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_6 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_57 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_156 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_59 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_47 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_7 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_178 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_67 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_53 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_8 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_73 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_200 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_75 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_59 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_9 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_81 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_222 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_83 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_65 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_10 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_89 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_244 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_71 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    tanh_11 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cuda:0', dtype=torch.float32)
    view_264 = rand_strided((512, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    mul_97 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.float32)
    view_266 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    sub_39 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    sub_41 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_4 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_6 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_134 = rand_strided((2, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((16384, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_152 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_153 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_154 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_185 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_186 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_217 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_284 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_285 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_286 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_316 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_317 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_318 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_319 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_349 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_351 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_352 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_382 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_384 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_385 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_416 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_417 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_449 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_450 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_451 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_481 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_482 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_49 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_483 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_484 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_514 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_515 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_51 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_516 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_517 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_16, primals_22, primals_28, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, sub_39, ne, sub_41, ne_3, ne_6, where_4, ne_8, where_6, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_151, permute_152, alias_29, permute_153, permute_154, permute_159, permute_163, permute_167, div_33, div_34, permute_184, permute_185, alias_31, permute_186, permute_187, div_36, div_37, permute_217, permute_218, alias_33, permute_219, permute_220, div_39, div_40, permute_250, permute_251, alias_35, permute_252, permute_253, div_42, div_43, permute_283, permute_284, alias_37, permute_285, permute_286, div_45, div_46, permute_316, permute_317, alias_39, permute_318, permute_319, div_48, div_49, permute_349, permute_350, alias_41, permute_351, permute_352, div_51, div_52, permute_382, permute_383, alias_43, permute_384, permute_385, div_54, div_55, permute_415, permute_416, alias_45, permute_417, permute_418, div_57, div_58, permute_448, permute_449, alias_47, permute_450, permute_451, div_60, div_61, permute_481, permute_482, alias_49, permute_483, permute_484, div_63, div_64, permute_514, permute_515, alias_51, permute_516, permute_517, permute_534, div_66, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AlbertForQuestionAnswering', benchmark_compiled_module)
