
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
# end_loss => convert_element_type_1, sum_29
# start_loss => convert_element_type, full_default_3, sum_26
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


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qyub65pv5jptuxi3z4qrle6umo7onuqcwjuq4cos2fxiokz65b.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp19 * tmp23
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ab/cabrdteljqx4zjzf5yypqg767fwd4slgkjqzbtqchrpjaqdpgnoy.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/lp/clp652zekwakrvu3zb54oqgaqfscnk2zxq7wptffew2xlyl5wflr.py
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
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqw3kkuwzfw4daweak4hclyak6imduqlek7lxg5c3kajmfrw53mx.py
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


# kernel path: /tmp/torchinductor_youkaichao/26/c267ylaj45yt2l6gkypo72a7upnqgtqmm7chali6emmqfdetg4x5.py
# Source Nodes: [intermediate_output_23], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_23 => add_192, erf_23, mul_167
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
    xnumel = 2097152
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


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdfujenr2tw3oyqqqzv6eznn7feymlij5uehnnwictodlohejo4.py
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


# kernel path: /tmp/torchinductor_youkaichao/67/c67oc26slmbqjpazd4dqkmiyqhtaowvec6vumzv6jx3swum6zha7.py
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


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2efhjuh4iefn3hg5l5obyljidxhc77y7avr4nocrb2rzqchoir.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 512
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 1024.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ia/ciacbwkx4xm3p3y7oategi5xoa4tdfxvjbfk6omyhhisquhc5w2m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 512
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
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp19 = 1024.0
    tmp20 = tmp6 * tmp19
    tmp21 = tmp20 - tmp10
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = tmp17 + tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.1111111111111112
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4vynu3ogqqg5fvfjbz5bxrheuwzzwktnltssi5y6wzpis45vk5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpddjq2fspsdhzozgh7vcyby5zwhjo22yqdxke5jk2qflrcgxvuu.py
# Source Nodes: [start_loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# start_loss => full_default_3
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*i1', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
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
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp38 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp19 = 1024.0
    tmp20 = tmp6 * tmp19
    tmp21 = tmp20 - tmp10
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = tmp17 + tmp24
    tmp27 = tl.full([1], -1, tl.int64)
    tmp28 = tmp26 == tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tmp33 = tmp25 * tmp32
    tmp34 = 0.0
    tmp35 = tl.where(tmp28, tmp34, tmp33)
    tmp36 = tl.full([1], False, tl.int1)
    tmp37 = tl.where(tmp36, tmp34, tmp33)
    tmp39 = tl.full([1], 0, tl.int64)
    tmp40 = tmp38 == tmp39
    tmp41 = tl.where(tmp40, tmp34, tmp33)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp37, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp41, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/su/csuc2jboh5cbe4q6o6fv6jblh3vososvcel7755rh56aiydenesi.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgx3pdu5vyhmht35qieahknnyuvzktyqlbb3utg7bblq4mjincdi.py
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
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2hmhxi7hzairwxghuxiwgje2qo7pxkquneokg3ahturx2mtz6kg.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29753344
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
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_206, primals_212, primals_222, primals_228, primals_238, primals_244, primals_254, primals_260, primals_270, primals_276, primals_286, primals_292, primals_302, primals_308, primals_318, primals_324, primals_334, primals_340, primals_350, primals_356, primals_366, primals_372, primals_382, primals_388, primals_393, full_default, slice_3, getitem_1, mul_1, view, clone_default_69, clone_default_70, clone_default_71, getitem_408, getitem_409, getitem_410, alias_default_47, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, clone_default_66, clone_default_67, clone_default_68, getitem_401, getitem_402, getitem_403, alias_default_45, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, clone_default_63, clone_default_64, clone_default_65, getitem_394, getitem_395, getitem_396, alias_default_43, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, clone_default_60, clone_default_61, clone_default_62, getitem_387, getitem_388, getitem_389, alias_default_41, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, clone_default_57, clone_default_58, clone_default_59, getitem_380, getitem_381, getitem_382, alias_default_39, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, clone_default_54, clone_default_55, clone_default_56, getitem_373, getitem_374, getitem_375, alias_default_37, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, clone_default_51, clone_default_52, clone_default_53, getitem_366, getitem_367, getitem_368, alias_default_35, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, clone_default_48, clone_default_49, clone_default_50, getitem_359, getitem_360, getitem_361, alias_default_33, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, clone_default_45, clone_default_46, clone_default_47, getitem_352, getitem_353, getitem_354, alias_default_31, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, clone_default_42, clone_default_43, clone_default_44, getitem_345, getitem_346, getitem_347, alias_default_29, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, clone_default_39, clone_default_40, clone_default_41, getitem_338, getitem_339, getitem_340, alias_default_27, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, clone_default_36, clone_default_37, clone_default_38, getitem_331, getitem_332, getitem_333, alias_default_25, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, view_264, clone_default_33, clone_default_34, clone_default_35, getitem_324, getitem_325, getitem_326, alias_default_23, view_280, getitem_127, mul_87, view_282, addmm_76, view_284, getitem_131, mul_92, view_286, clone_default_30, clone_default_31, clone_default_32, getitem_317, getitem_318, getitem_319, alias_default_21, view_302, getitem_137, mul_94, view_304, addmm_82, view_306, getitem_141, mul_99, view_308, clone_default_27, clone_default_28, clone_default_29, getitem_310, getitem_311, getitem_312, alias_default_19, view_324, getitem_147, mul_101, view_326, addmm_88, view_328, getitem_151, mul_106, view_330, clone_default_24, clone_default_25, clone_default_26, getitem_303, getitem_304, getitem_305, alias_default_17, view_346, getitem_157, mul_108, view_348, addmm_94, view_350, getitem_161, mul_113, view_352, clone_default_21, clone_default_22, clone_default_23, getitem_296, getitem_297, getitem_298, alias_default_15, view_368, getitem_167, mul_115, view_370, addmm_100, view_372, getitem_171, mul_120, view_374, clone_default_18, clone_default_19, clone_default_20, getitem_289, getitem_290, getitem_291, alias_default_13, view_390, getitem_177, mul_122, view_392, addmm_106, view_394, getitem_181, mul_127, view_396, clone_default_15, clone_default_16, clone_default_17, getitem_282, getitem_283, getitem_284, alias_default_11, view_412, getitem_187, mul_129, view_414, addmm_112, view_416, getitem_191, mul_134, view_418, clone_default_12, clone_default_13, clone_default_14, getitem_275, getitem_276, getitem_277, alias_default_9, view_434, getitem_197, mul_136, view_436, addmm_118, view_438, getitem_201, mul_141, view_440, clone_default_9, clone_default_10, clone_default_11, getitem_268, getitem_269, getitem_270, alias_default_7, view_456, getitem_207, mul_143, view_458, addmm_124, view_460, getitem_211, mul_148, view_462, clone_default_6, clone_default_7, clone_default_8, getitem_261, getitem_262, getitem_263, alias_default_5, view_478, getitem_217, mul_150, view_480, addmm_130, view_482, getitem_221, mul_155, view_484, clone_default_3, clone_default_4, clone_default_5, getitem_254, getitem_255, getitem_256, alias_default_3, view_500, getitem_227, mul_157, view_502, addmm_136, view_504, getitem_231, mul_162, view_506, clone_default, clone_default_1, clone_default_2, getitem_247, getitem_248, getitem_249, alias_default_1, view_522, getitem_237, mul_164, view_524, addmm_142, view_526, getitem_241, mul_169, view_528, sub_75, ne, sub_77, ne_3, ne_6, where_4, ne_8, where_6, permute_265, div_54, permute_269, permute_273, div_55, permute_277, permute_289, permute_294, permute_298, div_57, permute_302, permute_306, div_58, permute_310, permute_322, permute_327, permute_331, div_60, permute_335, permute_339, div_61, permute_343, permute_355, permute_360, permute_364, div_63, permute_368, permute_372, div_64, permute_376, permute_388, permute_393, permute_397, div_66, permute_401, permute_405, div_67, permute_409, permute_421, permute_426, permute_430, div_69, permute_434, permute_438, div_70, permute_442, permute_454, permute_459, permute_463, div_72, permute_467, permute_471, div_73, permute_475, permute_487, permute_492, permute_496, div_75, permute_500, permute_504, div_76, permute_508, permute_520, permute_525, permute_529, div_78, permute_533, permute_537, div_79, permute_541, permute_553, permute_558, permute_562, div_81, permute_566, permute_570, div_82, permute_574, permute_586, permute_591, permute_595, div_84, permute_599, permute_603, div_85, permute_607, permute_619, permute_624, permute_628, div_87, permute_632, permute_636, div_88, permute_640, permute_652, permute_657, permute_661, div_90, permute_665, permute_669, div_91, permute_673, permute_685, permute_690, permute_694, div_93, permute_698, permute_702, div_94, permute_706, permute_718, permute_723, permute_727, div_96, permute_731, permute_735, div_97, permute_739, permute_751, permute_756, permute_760, div_99, permute_764, permute_768, div_100, permute_772, permute_784, permute_789, permute_793, div_102, permute_797, permute_801, div_103, permute_805, permute_817, permute_822, permute_826, div_105, permute_830, permute_834, div_106, permute_838, permute_850, permute_855, permute_859, div_108, permute_863, permute_867, div_109, permute_871, permute_883, permute_888, permute_892, div_111, permute_896, permute_900, div_112, permute_904, permute_916, permute_921, permute_925, div_114, permute_929, permute_933, div_115, permute_937, permute_949, permute_954, permute_958, div_117, permute_962, permute_966, div_118, permute_970, permute_982, permute_987, permute_991, div_120, permute_995, permute_999, div_121, permute_1003, permute_1015, permute_1020, permute_1024, div_123, permute_1028, permute_1032, div_124, permute_1036, permute_1048, permute_1053, permute_1057, div_126, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_4, (1024, ), (1, ))
    assert_size_stride(primals_14, (1024, ), (1, ))
    assert_size_stride(primals_20, (1024, ), (1, ))
    assert_size_stride(primals_30, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, ), (1, ))
    assert_size_stride(primals_46, (1024, ), (1, ))
    assert_size_stride(primals_52, (1024, ), (1, ))
    assert_size_stride(primals_62, (1024, ), (1, ))
    assert_size_stride(primals_68, (1024, ), (1, ))
    assert_size_stride(primals_78, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_94, (1024, ), (1, ))
    assert_size_stride(primals_100, (1024, ), (1, ))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_116, (1024, ), (1, ))
    assert_size_stride(primals_126, (1024, ), (1, ))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_142, (1024, ), (1, ))
    assert_size_stride(primals_148, (1024, ), (1, ))
    assert_size_stride(primals_158, (1024, ), (1, ))
    assert_size_stride(primals_164, (1024, ), (1, ))
    assert_size_stride(primals_174, (1024, ), (1, ))
    assert_size_stride(primals_180, (1024, ), (1, ))
    assert_size_stride(primals_190, (1024, ), (1, ))
    assert_size_stride(primals_196, (1024, ), (1, ))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_212, (1024, ), (1, ))
    assert_size_stride(primals_222, (1024, ), (1, ))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_238, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_254, (1024, ), (1, ))
    assert_size_stride(primals_260, (1024, ), (1, ))
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_276, (1024, ), (1, ))
    assert_size_stride(primals_286, (1024, ), (1, ))
    assert_size_stride(primals_292, (1024, ), (1, ))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_308, (1024, ), (1, ))
    assert_size_stride(primals_318, (1024, ), (1, ))
    assert_size_stride(primals_324, (1024, ), (1, ))
    assert_size_stride(primals_334, (1024, ), (1, ))
    assert_size_stride(primals_340, (1024, ), (1, ))
    assert_size_stride(primals_350, (1024, ), (1, ))
    assert_size_stride(primals_356, (1024, ), (1, ))
    assert_size_stride(primals_366, (1024, ), (1, ))
    assert_size_stride(primals_372, (1024, ), (1, ))
    assert_size_stride(primals_382, (1024, ), (1, ))
    assert_size_stride(primals_388, (1024, ), (1, ))
    assert_size_stride(primals_393, (1, 512), (512, 1))
    assert_size_stride(full_default, (1, 512), (512, 1))
    assert_size_stride(slice_3, (1, 512), (512, 1))
    assert_size_stride(getitem_1, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_1, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_69, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_70, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_71, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_408, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_409, (), ())
    assert_size_stride(getitem_410, (), ())
    assert_size_stride(alias_default_47, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_16, (512, 1024), (1024, 1))
    assert_size_stride(getitem_7, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_3, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_18, (512, 1024), (1024, 1))
    assert_size_stride(addmm_4, (512, 4096), (4096, 1))
    assert_size_stride(view_20, (512, 4096), (4096, 1))
    assert_size_stride(getitem_11, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_8, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_22, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_66, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_67, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_68, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_401, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_402, (), ())
    assert_size_stride(getitem_403, (), ())
    assert_size_stride(alias_default_45, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_38, (512, 1024), (1024, 1))
    assert_size_stride(getitem_17, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_10, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_40, (512, 1024), (1024, 1))
    assert_size_stride(addmm_10, (512, 4096), (4096, 1))
    assert_size_stride(view_42, (512, 4096), (4096, 1))
    assert_size_stride(getitem_21, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_15, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_44, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_63, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_64, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_65, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_394, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_395, (), ())
    assert_size_stride(getitem_396, (), ())
    assert_size_stride(alias_default_43, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_60, (512, 1024), (1024, 1))
    assert_size_stride(getitem_27, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_17, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_62, (512, 1024), (1024, 1))
    assert_size_stride(addmm_16, (512, 4096), (4096, 1))
    assert_size_stride(view_64, (512, 4096), (4096, 1))
    assert_size_stride(getitem_31, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_22, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_66, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_60, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_61, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_62, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_387, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_388, (), ())
    assert_size_stride(getitem_389, (), ())
    assert_size_stride(alias_default_41, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_82, (512, 1024), (1024, 1))
    assert_size_stride(getitem_37, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_24, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_84, (512, 1024), (1024, 1))
    assert_size_stride(addmm_22, (512, 4096), (4096, 1))
    assert_size_stride(view_86, (512, 4096), (4096, 1))
    assert_size_stride(getitem_41, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_29, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_88, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_57, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_58, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_59, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_380, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_381, (), ())
    assert_size_stride(getitem_382, (), ())
    assert_size_stride(alias_default_39, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_104, (512, 1024), (1024, 1))
    assert_size_stride(getitem_47, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_31, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_106, (512, 1024), (1024, 1))
    assert_size_stride(addmm_28, (512, 4096), (4096, 1))
    assert_size_stride(view_108, (512, 4096), (4096, 1))
    assert_size_stride(getitem_51, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_36, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_110, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_54, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_55, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_56, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_373, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_374, (), ())
    assert_size_stride(getitem_375, (), ())
    assert_size_stride(alias_default_37, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_126, (512, 1024), (1024, 1))
    assert_size_stride(getitem_57, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_38, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_128, (512, 1024), (1024, 1))
    assert_size_stride(addmm_34, (512, 4096), (4096, 1))
    assert_size_stride(view_130, (512, 4096), (4096, 1))
    assert_size_stride(getitem_61, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_43, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_132, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_51, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_52, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_53, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_366, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_367, (), ())
    assert_size_stride(getitem_368, (), ())
    assert_size_stride(alias_default_35, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_148, (512, 1024), (1024, 1))
    assert_size_stride(getitem_67, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_45, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_150, (512, 1024), (1024, 1))
    assert_size_stride(addmm_40, (512, 4096), (4096, 1))
    assert_size_stride(view_152, (512, 4096), (4096, 1))
    assert_size_stride(getitem_71, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_50, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_154, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_48, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_49, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_50, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_359, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_360, (), ())
    assert_size_stride(getitem_361, (), ())
    assert_size_stride(alias_default_33, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_170, (512, 1024), (1024, 1))
    assert_size_stride(getitem_77, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_52, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_172, (512, 1024), (1024, 1))
    assert_size_stride(addmm_46, (512, 4096), (4096, 1))
    assert_size_stride(view_174, (512, 4096), (4096, 1))
    assert_size_stride(getitem_81, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_57, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_176, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_45, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_46, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_47, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_352, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_353, (), ())
    assert_size_stride(getitem_354, (), ())
    assert_size_stride(alias_default_31, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_192, (512, 1024), (1024, 1))
    assert_size_stride(getitem_87, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_59, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_194, (512, 1024), (1024, 1))
    assert_size_stride(addmm_52, (512, 4096), (4096, 1))
    assert_size_stride(view_196, (512, 4096), (4096, 1))
    assert_size_stride(getitem_91, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_64, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_198, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_42, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_43, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_44, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_345, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_346, (), ())
    assert_size_stride(getitem_347, (), ())
    assert_size_stride(alias_default_29, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_214, (512, 1024), (1024, 1))
    assert_size_stride(getitem_97, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_66, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_216, (512, 1024), (1024, 1))
    assert_size_stride(addmm_58, (512, 4096), (4096, 1))
    assert_size_stride(view_218, (512, 4096), (4096, 1))
    assert_size_stride(getitem_101, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_71, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_220, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_39, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_40, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_41, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_338, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_339, (), ())
    assert_size_stride(getitem_340, (), ())
    assert_size_stride(alias_default_27, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_236, (512, 1024), (1024, 1))
    assert_size_stride(getitem_107, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_73, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_238, (512, 1024), (1024, 1))
    assert_size_stride(addmm_64, (512, 4096), (4096, 1))
    assert_size_stride(view_240, (512, 4096), (4096, 1))
    assert_size_stride(getitem_111, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_78, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_242, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_36, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_37, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_38, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_331, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_332, (), ())
    assert_size_stride(getitem_333, (), ())
    assert_size_stride(alias_default_25, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_258, (512, 1024), (1024, 1))
    assert_size_stride(getitem_117, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_80, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_260, (512, 1024), (1024, 1))
    assert_size_stride(addmm_70, (512, 4096), (4096, 1))
    assert_size_stride(view_262, (512, 4096), (4096, 1))
    assert_size_stride(getitem_121, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_85, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_264, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_33, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_34, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_35, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_324, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_325, (), ())
    assert_size_stride(getitem_326, (), ())
    assert_size_stride(alias_default_23, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_280, (512, 1024), (1024, 1))
    assert_size_stride(getitem_127, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_87, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_282, (512, 1024), (1024, 1))
    assert_size_stride(addmm_76, (512, 4096), (4096, 1))
    assert_size_stride(view_284, (512, 4096), (4096, 1))
    assert_size_stride(getitem_131, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_92, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_286, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_30, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_31, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_32, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_317, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_318, (), ())
    assert_size_stride(getitem_319, (), ())
    assert_size_stride(alias_default_21, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_302, (512, 1024), (1024, 1))
    assert_size_stride(getitem_137, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_94, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_304, (512, 1024), (1024, 1))
    assert_size_stride(addmm_82, (512, 4096), (4096, 1))
    assert_size_stride(view_306, (512, 4096), (4096, 1))
    assert_size_stride(getitem_141, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_99, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_308, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_27, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_28, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_29, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_310, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_311, (), ())
    assert_size_stride(getitem_312, (), ())
    assert_size_stride(alias_default_19, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_324, (512, 1024), (1024, 1))
    assert_size_stride(getitem_147, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_101, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_326, (512, 1024), (1024, 1))
    assert_size_stride(addmm_88, (512, 4096), (4096, 1))
    assert_size_stride(view_328, (512, 4096), (4096, 1))
    assert_size_stride(getitem_151, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_106, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_330, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_24, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_25, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_26, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_303, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_304, (), ())
    assert_size_stride(getitem_305, (), ())
    assert_size_stride(alias_default_17, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_346, (512, 1024), (1024, 1))
    assert_size_stride(getitem_157, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_108, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_348, (512, 1024), (1024, 1))
    assert_size_stride(addmm_94, (512, 4096), (4096, 1))
    assert_size_stride(view_350, (512, 4096), (4096, 1))
    assert_size_stride(getitem_161, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_113, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_352, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_21, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_22, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_23, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_296, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_297, (), ())
    assert_size_stride(getitem_298, (), ())
    assert_size_stride(alias_default_15, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_368, (512, 1024), (1024, 1))
    assert_size_stride(getitem_167, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_115, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_370, (512, 1024), (1024, 1))
    assert_size_stride(addmm_100, (512, 4096), (4096, 1))
    assert_size_stride(view_372, (512, 4096), (4096, 1))
    assert_size_stride(getitem_171, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_120, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_374, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_18, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_19, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_20, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_289, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_290, (), ())
    assert_size_stride(getitem_291, (), ())
    assert_size_stride(alias_default_13, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_390, (512, 1024), (1024, 1))
    assert_size_stride(getitem_177, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_122, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_392, (512, 1024), (1024, 1))
    assert_size_stride(addmm_106, (512, 4096), (4096, 1))
    assert_size_stride(view_394, (512, 4096), (4096, 1))
    assert_size_stride(getitem_181, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_127, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_396, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_15, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_16, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_17, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_282, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_283, (), ())
    assert_size_stride(getitem_284, (), ())
    assert_size_stride(alias_default_11, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_412, (512, 1024), (1024, 1))
    assert_size_stride(getitem_187, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_129, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_414, (512, 1024), (1024, 1))
    assert_size_stride(addmm_112, (512, 4096), (4096, 1))
    assert_size_stride(view_416, (512, 4096), (4096, 1))
    assert_size_stride(getitem_191, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_134, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_418, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_12, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_13, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_14, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_275, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_276, (), ())
    assert_size_stride(getitem_277, (), ())
    assert_size_stride(alias_default_9, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_434, (512, 1024), (1024, 1))
    assert_size_stride(getitem_197, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_136, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_436, (512, 1024), (1024, 1))
    assert_size_stride(addmm_118, (512, 4096), (4096, 1))
    assert_size_stride(view_438, (512, 4096), (4096, 1))
    assert_size_stride(getitem_201, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_141, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_440, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_9, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_10, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_11, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_268, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_269, (), ())
    assert_size_stride(getitem_270, (), ())
    assert_size_stride(alias_default_7, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_456, (512, 1024), (1024, 1))
    assert_size_stride(getitem_207, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_143, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_458, (512, 1024), (1024, 1))
    assert_size_stride(addmm_124, (512, 4096), (4096, 1))
    assert_size_stride(view_460, (512, 4096), (4096, 1))
    assert_size_stride(getitem_211, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_148, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_462, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_6, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_7, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_8, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_261, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_262, (), ())
    assert_size_stride(getitem_263, (), ())
    assert_size_stride(alias_default_5, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_478, (512, 1024), (1024, 1))
    assert_size_stride(getitem_217, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_150, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_480, (512, 1024), (1024, 1))
    assert_size_stride(addmm_130, (512, 4096), (4096, 1))
    assert_size_stride(view_482, (512, 4096), (4096, 1))
    assert_size_stride(getitem_221, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_155, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_484, (512, 1024), (1024, 1))
    assert_size_stride(clone_default_3, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_4, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_5, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_254, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_255, (), ())
    assert_size_stride(getitem_256, (), ())
    assert_size_stride(alias_default_3, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_500, (512, 1024), (1024, 1))
    assert_size_stride(getitem_227, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_157, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_502, (512, 1024), (1024, 1))
    assert_size_stride(addmm_136, (512, 4096), (4096, 1))
    assert_size_stride(view_504, (512, 4096), (4096, 1))
    assert_size_stride(getitem_231, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_162, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_506, (512, 1024), (1024, 1))
    assert_size_stride(clone_default, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_1, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(clone_default_2, (1, 16, 512, 64), (524288, 32768, 64, 1))
    assert_size_stride(getitem_247, (1, 16, 512), (8192, 512, 1))
    assert_size_stride(getitem_248, (), ())
    assert_size_stride(getitem_249, (), ())
    assert_size_stride(alias_default_1, (1, 16, 512, 64), (524288, 64, 1024, 1))
    assert_size_stride(view_522, (512, 1024), (1024, 1))
    assert_size_stride(getitem_237, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_164, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_524, (512, 1024), (1024, 1))
    assert_size_stride(addmm_142, (512, 4096), (4096, 1))
    assert_size_stride(view_526, (512, 4096), (4096, 1))
    assert_size_stride(getitem_241, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_169, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_528, (512, 1024), (1024, 1))
    assert_size_stride(sub_75, (1, 512), (512, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_77, (1, 512), (512, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_4, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_6, (1, 1), (1, 1))
    assert_size_stride(permute_265, (2, 1024), (1024, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_269, (1024, 4096), (4096, 1))
    assert_size_stride(permute_273, (4096, 1024), (1024, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_277, (1024, 1024), (1024, 1))
    assert_size_stride(permute_289, (1024, 1024), (1024, 1))
    assert_size_stride(permute_294, (1024, 1024), (1024, 1))
    assert_size_stride(permute_298, (1024, 1024), (1024, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_302, (1024, 4096), (4096, 1))
    assert_size_stride(permute_306, (4096, 1024), (1024, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_310, (1024, 1024), (1024, 1))
    assert_size_stride(permute_322, (1024, 1024), (1024, 1))
    assert_size_stride(permute_327, (1024, 1024), (1024, 1))
    assert_size_stride(permute_331, (1024, 1024), (1024, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_335, (1024, 4096), (4096, 1))
    assert_size_stride(permute_339, (4096, 1024), (1024, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_343, (1024, 1024), (1024, 1))
    assert_size_stride(permute_355, (1024, 1024), (1024, 1))
    assert_size_stride(permute_360, (1024, 1024), (1024, 1))
    assert_size_stride(permute_364, (1024, 1024), (1024, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_368, (1024, 4096), (4096, 1))
    assert_size_stride(permute_372, (4096, 1024), (1024, 1))
    assert_size_stride(div_64, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_376, (1024, 1024), (1024, 1))
    assert_size_stride(permute_388, (1024, 1024), (1024, 1))
    assert_size_stride(permute_393, (1024, 1024), (1024, 1))
    assert_size_stride(permute_397, (1024, 1024), (1024, 1))
    assert_size_stride(div_66, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_401, (1024, 4096), (4096, 1))
    assert_size_stride(permute_405, (4096, 1024), (1024, 1))
    assert_size_stride(div_67, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_409, (1024, 1024), (1024, 1))
    assert_size_stride(permute_421, (1024, 1024), (1024, 1))
    assert_size_stride(permute_426, (1024, 1024), (1024, 1))
    assert_size_stride(permute_430, (1024, 1024), (1024, 1))
    assert_size_stride(div_69, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_434, (1024, 4096), (4096, 1))
    assert_size_stride(permute_438, (4096, 1024), (1024, 1))
    assert_size_stride(div_70, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_442, (1024, 1024), (1024, 1))
    assert_size_stride(permute_454, (1024, 1024), (1024, 1))
    assert_size_stride(permute_459, (1024, 1024), (1024, 1))
    assert_size_stride(permute_463, (1024, 1024), (1024, 1))
    assert_size_stride(div_72, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_467, (1024, 4096), (4096, 1))
    assert_size_stride(permute_471, (4096, 1024), (1024, 1))
    assert_size_stride(div_73, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_475, (1024, 1024), (1024, 1))
    assert_size_stride(permute_487, (1024, 1024), (1024, 1))
    assert_size_stride(permute_492, (1024, 1024), (1024, 1))
    assert_size_stride(permute_496, (1024, 1024), (1024, 1))
    assert_size_stride(div_75, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_500, (1024, 4096), (4096, 1))
    assert_size_stride(permute_504, (4096, 1024), (1024, 1))
    assert_size_stride(div_76, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_508, (1024, 1024), (1024, 1))
    assert_size_stride(permute_520, (1024, 1024), (1024, 1))
    assert_size_stride(permute_525, (1024, 1024), (1024, 1))
    assert_size_stride(permute_529, (1024, 1024), (1024, 1))
    assert_size_stride(div_78, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_533, (1024, 4096), (4096, 1))
    assert_size_stride(permute_537, (4096, 1024), (1024, 1))
    assert_size_stride(div_79, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_541, (1024, 1024), (1024, 1))
    assert_size_stride(permute_553, (1024, 1024), (1024, 1))
    assert_size_stride(permute_558, (1024, 1024), (1024, 1))
    assert_size_stride(permute_562, (1024, 1024), (1024, 1))
    assert_size_stride(div_81, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_566, (1024, 4096), (4096, 1))
    assert_size_stride(permute_570, (4096, 1024), (1024, 1))
    assert_size_stride(div_82, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_574, (1024, 1024), (1024, 1))
    assert_size_stride(permute_586, (1024, 1024), (1024, 1))
    assert_size_stride(permute_591, (1024, 1024), (1024, 1))
    assert_size_stride(permute_595, (1024, 1024), (1024, 1))
    assert_size_stride(div_84, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_599, (1024, 4096), (4096, 1))
    assert_size_stride(permute_603, (4096, 1024), (1024, 1))
    assert_size_stride(div_85, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_607, (1024, 1024), (1024, 1))
    assert_size_stride(permute_619, (1024, 1024), (1024, 1))
    assert_size_stride(permute_624, (1024, 1024), (1024, 1))
    assert_size_stride(permute_628, (1024, 1024), (1024, 1))
    assert_size_stride(div_87, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_632, (1024, 4096), (4096, 1))
    assert_size_stride(permute_636, (4096, 1024), (1024, 1))
    assert_size_stride(div_88, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_640, (1024, 1024), (1024, 1))
    assert_size_stride(permute_652, (1024, 1024), (1024, 1))
    assert_size_stride(permute_657, (1024, 1024), (1024, 1))
    assert_size_stride(permute_661, (1024, 1024), (1024, 1))
    assert_size_stride(div_90, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_665, (1024, 4096), (4096, 1))
    assert_size_stride(permute_669, (4096, 1024), (1024, 1))
    assert_size_stride(div_91, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_673, (1024, 1024), (1024, 1))
    assert_size_stride(permute_685, (1024, 1024), (1024, 1))
    assert_size_stride(permute_690, (1024, 1024), (1024, 1))
    assert_size_stride(permute_694, (1024, 1024), (1024, 1))
    assert_size_stride(div_93, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_698, (1024, 4096), (4096, 1))
    assert_size_stride(permute_702, (4096, 1024), (1024, 1))
    assert_size_stride(div_94, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_706, (1024, 1024), (1024, 1))
    assert_size_stride(permute_718, (1024, 1024), (1024, 1))
    assert_size_stride(permute_723, (1024, 1024), (1024, 1))
    assert_size_stride(permute_727, (1024, 1024), (1024, 1))
    assert_size_stride(div_96, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_731, (1024, 4096), (4096, 1))
    assert_size_stride(permute_735, (4096, 1024), (1024, 1))
    assert_size_stride(div_97, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_739, (1024, 1024), (1024, 1))
    assert_size_stride(permute_751, (1024, 1024), (1024, 1))
    assert_size_stride(permute_756, (1024, 1024), (1024, 1))
    assert_size_stride(permute_760, (1024, 1024), (1024, 1))
    assert_size_stride(div_99, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_764, (1024, 4096), (4096, 1))
    assert_size_stride(permute_768, (4096, 1024), (1024, 1))
    assert_size_stride(div_100, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_772, (1024, 1024), (1024, 1))
    assert_size_stride(permute_784, (1024, 1024), (1024, 1))
    assert_size_stride(permute_789, (1024, 1024), (1024, 1))
    assert_size_stride(permute_793, (1024, 1024), (1024, 1))
    assert_size_stride(div_102, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_797, (1024, 4096), (4096, 1))
    assert_size_stride(permute_801, (4096, 1024), (1024, 1))
    assert_size_stride(div_103, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_805, (1024, 1024), (1024, 1))
    assert_size_stride(permute_817, (1024, 1024), (1024, 1))
    assert_size_stride(permute_822, (1024, 1024), (1024, 1))
    assert_size_stride(permute_826, (1024, 1024), (1024, 1))
    assert_size_stride(div_105, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_830, (1024, 4096), (4096, 1))
    assert_size_stride(permute_834, (4096, 1024), (1024, 1))
    assert_size_stride(div_106, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_838, (1024, 1024), (1024, 1))
    assert_size_stride(permute_850, (1024, 1024), (1024, 1))
    assert_size_stride(permute_855, (1024, 1024), (1024, 1))
    assert_size_stride(permute_859, (1024, 1024), (1024, 1))
    assert_size_stride(div_108, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_863, (1024, 4096), (4096, 1))
    assert_size_stride(permute_867, (4096, 1024), (1024, 1))
    assert_size_stride(div_109, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_871, (1024, 1024), (1024, 1))
    assert_size_stride(permute_883, (1024, 1024), (1024, 1))
    assert_size_stride(permute_888, (1024, 1024), (1024, 1))
    assert_size_stride(permute_892, (1024, 1024), (1024, 1))
    assert_size_stride(div_111, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_896, (1024, 4096), (4096, 1))
    assert_size_stride(permute_900, (4096, 1024), (1024, 1))
    assert_size_stride(div_112, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_904, (1024, 1024), (1024, 1))
    assert_size_stride(permute_916, (1024, 1024), (1024, 1))
    assert_size_stride(permute_921, (1024, 1024), (1024, 1))
    assert_size_stride(permute_925, (1024, 1024), (1024, 1))
    assert_size_stride(div_114, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_929, (1024, 4096), (4096, 1))
    assert_size_stride(permute_933, (4096, 1024), (1024, 1))
    assert_size_stride(div_115, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_937, (1024, 1024), (1024, 1))
    assert_size_stride(permute_949, (1024, 1024), (1024, 1))
    assert_size_stride(permute_954, (1024, 1024), (1024, 1))
    assert_size_stride(permute_958, (1024, 1024), (1024, 1))
    assert_size_stride(div_117, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_962, (1024, 4096), (4096, 1))
    assert_size_stride(permute_966, (4096, 1024), (1024, 1))
    assert_size_stride(div_118, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_970, (1024, 1024), (1024, 1))
    assert_size_stride(permute_982, (1024, 1024), (1024, 1))
    assert_size_stride(permute_987, (1024, 1024), (1024, 1))
    assert_size_stride(permute_991, (1024, 1024), (1024, 1))
    assert_size_stride(div_120, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_995, (1024, 4096), (4096, 1))
    assert_size_stride(permute_999, (4096, 1024), (1024, 1))
    assert_size_stride(div_121, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_1003, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1015, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1020, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1024, (1024, 1024), (1024, 1))
    assert_size_stride(div_123, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_1028, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1032, (4096, 1024), (1024, 1))
    assert_size_stride(div_124, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_1036, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1048, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1053, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1057, (1024, 1024), (1024, 1))
    assert_size_stride(div_126, (1, 512, 1), (512, 1, 1))
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
        triton_poi_fused_cat_2.run(tangents_2, buf4, ne_8, tangents_1, ne, sub_75, buf7, tangents_3, buf0, ne_6, ne_3, sub_77, buf3, buf8, 1024, grid=grid(1024), stream=stream0)
        del buf0
        del buf3
        del buf4
        del buf7
        del ne
        del ne_3
        del ne_6
        del ne_8
        del sub_75
        del sub_77
        del tangents_1
        del tangents_2
        del tangents_3
        buf9 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (512, 2), (2, 1), 0), permute_265, out=buf9)
        del permute_265
        buf10 = empty((2, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (2, 512), (1, 2), 0), view_528, out=buf10)
        del view_528
        buf11 = empty_strided((1, 2, 4), (8, 1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf8, buf11, 8, 128, grid=grid(8), stream=stream0)
        buf12 = empty((1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf11, buf12, 2, 4, grid=grid(2), stream=stream0)
        del buf11
        buf15 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        buf18 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_5.run(buf9, primals_388, mul_169, div_54, getitem_241, buf15, buf18, 512, 1024, grid=grid(512), stream=stream0)
        del div_54
        del getitem_241
        del primals_388
        buf16 = reinterpret_tensor(buf8, (1024, ), (1, ), 0); del buf8  # reuse
        buf17 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf9, mul_169, buf16, buf17, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_169
        buf19 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (512, 1024), (1024, 1), 0), permute_269, out=buf19)
        del permute_269
        buf20 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (1024, 512), (1, 1024), 0), view_526, out=buf20)
        del view_526
        buf21 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf18, buf21, 4096, 128, grid=grid(4096), stream=stream0)
        buf22 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf21, buf22, 1024, 4, grid=grid(1024), stream=stream0)
        buf23 = reinterpret_tensor(buf19, (1, 512, 4096), (2097152, 4096, 1), 0); del buf19  # reuse
        # Source Nodes: [intermediate_output_23], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf23, addmm_142, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_142
        buf24 = reinterpret_tensor(buf18, (512, 1024), (1024, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (512, 4096), (4096, 1), 0), permute_273, out=buf24)
        del permute_273
        buf25 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (4096, 512), (1, 4096), 0), view_524, out=buf25)
        del view_524
        buf26 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf23, buf26, 16384, 128, grid=grid(16384), stream=stream0)
        buf27 = reinterpret_tensor(buf21, (1, 4096), (4096, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf26, buf27, 4096, 4, grid=grid(4096), stream=stream0)
        buf32 = buf15; del buf15  # reuse
        buf33 = reinterpret_tensor(buf9, (1, 512, 1024), (524288, 1024, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf32, buf24, primals_382, mul_164, div_55, getitem_237, buf33, 512, 1024, grid=grid(512), stream=stream0)
        del div_55
        del getitem_237
        del primals_382
        buf30 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf31 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf24, mul_164, buf30, buf31, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_164
        buf34 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (512, 1024), (1024, 1), 0), permute_277, out=buf34)
        del permute_277
        buf35 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (1024, 512), (1, 1024), 0), view_522, out=buf35)
        del view_522
        buf36 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf33, buf36, 4096, 128, grid=grid(4096), stream=stream0)
        del buf33
        buf37 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf36, buf37, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf38 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf34, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_247, getitem_248, getitem_249, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_1
        del clone_default
        del clone_default_1
        del clone_default_2
        del getitem_247
        del getitem_248
        del getitem_249
        buf39 = buf38[0]
        buf40 = buf38[1]
        buf41 = buf38[2]
        del buf38
        buf42 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (512, 1024), (1024, 1), 0), permute_289, out=buf42)
        del permute_289
        buf43 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (1024, 512), (1, 1024), 0), view_506, out=buf43)
        buf44 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf41, buf44, 4096, 128, grid=grid(4096), stream=stream0)
        buf45 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf44, buf45, 1024, 4, grid=grid(1024), stream=stream0)
        buf46 = reinterpret_tensor(buf41, (512, 1024), (1024, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (512, 1024), (1024, 1), 0), permute_294, out=buf46)
        del permute_294
        buf47 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (1024, 512), (1, 1024), 0), view_506, out=buf47)
        buf48 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf40, buf48, 4096, 128, grid=grid(4096), stream=stream0)
        buf49 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf48, buf49, 1024, 4, grid=grid(1024), stream=stream0)
        buf50 = reinterpret_tensor(buf40, (512, 1024), (1024, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (512, 1024), (1024, 1), 0), permute_298, out=buf50)
        del permute_298
        buf51 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (1024, 512), (1, 1024), 0), view_506, out=buf51)
        del view_506
        buf52 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf39, buf52, 4096, 128, grid=grid(4096), stream=stream0)
        buf53 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf52, buf53, 1024, 4, grid=grid(1024), stream=stream0)
        buf58 = buf32; del buf32  # reuse
        buf59 = reinterpret_tensor(buf39, (1, 512, 1024), (524288, 1024, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf58, buf42, buf46, buf50, primals_372, mul_162, div_57, getitem_231, buf59, 512, 1024, grid=grid(512), stream=stream0)
        del div_57
        del getitem_231
        del primals_372
        buf56 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf57 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf42, buf46, buf50, mul_162, buf56, buf57, 1024, 512, grid=grid(1024), stream=stream0)
        del buf42
        del buf46
        del mul_162
        buf60 = reinterpret_tensor(buf23, (512, 4096), (4096, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (512, 1024), (1024, 1), 0), permute_302, out=buf60)
        del permute_302
        buf61 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (1024, 512), (1, 1024), 0), view_504, out=buf61)
        del view_504
        buf62 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf59, buf62, 4096, 128, grid=grid(4096), stream=stream0)
        buf63 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf62, buf63, 1024, 4, grid=grid(1024), stream=stream0)
        buf64 = reinterpret_tensor(buf60, (1, 512, 4096), (2097152, 4096, 1), 0); del buf60  # reuse
        # Source Nodes: [intermediate_output_22], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf64, addmm_136, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_136
        buf65 = reinterpret_tensor(buf59, (512, 1024), (1024, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 4096), (4096, 1), 0), permute_306, out=buf65)
        del permute_306
        buf66 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (4096, 512), (1, 4096), 0), view_502, out=buf66)
        del view_502
        buf67 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf64, buf67, 16384, 128, grid=grid(16384), stream=stream0)
        buf68 = reinterpret_tensor(buf62, (1, 4096), (4096, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf67, buf68, 4096, 4, grid=grid(4096), stream=stream0)
        buf73 = buf58; del buf58  # reuse
        buf74 = reinterpret_tensor(buf50, (1, 512, 1024), (524288, 1024, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf73, buf65, primals_366, mul_157, div_58, getitem_227, buf74, 512, 1024, grid=grid(512), stream=stream0)
        del div_58
        del getitem_227
        del primals_366
        buf71 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf72 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf65, mul_157, buf71, buf72, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_157
        buf75 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 1024), (1024, 1), 0), permute_310, out=buf75)
        del permute_310
        buf76 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (1024, 512), (1, 1024), 0), view_500, out=buf76)
        del view_500
        buf77 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf74, buf77, 4096, 128, grid=grid(4096), stream=stream0)
        del buf74
        buf78 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf77, buf78, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf79 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf75, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_254, getitem_255, getitem_256, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_3
        del clone_default_3
        del clone_default_4
        del clone_default_5
        del getitem_254
        del getitem_255
        del getitem_256
        buf80 = buf79[0]
        buf81 = buf79[1]
        buf82 = buf79[2]
        del buf79
        buf83 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (512, 1024), (1024, 1), 0), permute_322, out=buf83)
        del permute_322
        buf84 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (1024, 512), (1, 1024), 0), view_484, out=buf84)
        buf85 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf82, buf85, 4096, 128, grid=grid(4096), stream=stream0)
        buf86 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf85, buf86, 1024, 4, grid=grid(1024), stream=stream0)
        buf87 = reinterpret_tensor(buf82, (512, 1024), (1024, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (512, 1024), (1024, 1), 0), permute_327, out=buf87)
        del permute_327
        buf88 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (1024, 512), (1, 1024), 0), view_484, out=buf88)
        buf89 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf81, buf89, 4096, 128, grid=grid(4096), stream=stream0)
        buf90 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf89, buf90, 1024, 4, grid=grid(1024), stream=stream0)
        buf91 = reinterpret_tensor(buf81, (512, 1024), (1024, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (512, 1024), (1024, 1), 0), permute_331, out=buf91)
        del permute_331
        buf92 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (1024, 512), (1, 1024), 0), view_484, out=buf92)
        del view_484
        buf93 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf80, buf93, 4096, 128, grid=grid(4096), stream=stream0)
        buf94 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf93, buf94, 1024, 4, grid=grid(1024), stream=stream0)
        buf99 = buf73; del buf73  # reuse
        buf100 = reinterpret_tensor(buf80, (1, 512, 1024), (524288, 1024, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf99, buf83, buf87, buf91, primals_356, mul_155, div_60, getitem_221, buf100, 512, 1024, grid=grid(512), stream=stream0)
        del div_60
        del getitem_221
        del primals_356
        buf97 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf98 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf83, buf87, buf91, mul_155, buf97, buf98, 1024, 512, grid=grid(1024), stream=stream0)
        del buf83
        del buf87
        del mul_155
        buf101 = reinterpret_tensor(buf64, (512, 4096), (4096, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (512, 1024), (1024, 1), 0), permute_335, out=buf101)
        del permute_335
        buf102 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (1024, 512), (1, 1024), 0), view_482, out=buf102)
        del view_482
        buf103 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf100, buf103, 4096, 128, grid=grid(4096), stream=stream0)
        buf104 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf103, buf104, 1024, 4, grid=grid(1024), stream=stream0)
        buf105 = reinterpret_tensor(buf101, (1, 512, 4096), (2097152, 4096, 1), 0); del buf101  # reuse
        # Source Nodes: [intermediate_output_21], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf105, addmm_130, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_130
        buf106 = reinterpret_tensor(buf100, (512, 1024), (1024, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (512, 4096), (4096, 1), 0), permute_339, out=buf106)
        del permute_339
        buf107 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (4096, 512), (1, 4096), 0), view_480, out=buf107)
        del view_480
        buf108 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf105, buf108, 16384, 128, grid=grid(16384), stream=stream0)
        buf109 = reinterpret_tensor(buf103, (1, 4096), (4096, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf108, buf109, 4096, 4, grid=grid(4096), stream=stream0)
        buf114 = buf99; del buf99  # reuse
        buf115 = reinterpret_tensor(buf91, (1, 512, 1024), (524288, 1024, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf114, buf106, primals_350, mul_150, div_61, getitem_217, buf115, 512, 1024, grid=grid(512), stream=stream0)
        del div_61
        del getitem_217
        del primals_350
        buf112 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf113 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf106, mul_150, buf112, buf113, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_150
        buf116 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (512, 1024), (1024, 1), 0), permute_343, out=buf116)
        del permute_343
        buf117 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (1024, 512), (1, 1024), 0), view_478, out=buf117)
        del view_478
        buf118 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf115, buf118, 4096, 128, grid=grid(4096), stream=stream0)
        del buf115
        buf119 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf118, buf119, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf120 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf116, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_261, getitem_262, getitem_263, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_5
        del clone_default_6
        del clone_default_7
        del clone_default_8
        del getitem_261
        del getitem_262
        del getitem_263
        buf121 = buf120[0]
        buf122 = buf120[1]
        buf123 = buf120[2]
        del buf120
        buf124 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (512, 1024), (1024, 1), 0), permute_355, out=buf124)
        del permute_355
        buf125 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 512), (1, 1024), 0), view_462, out=buf125)
        buf126 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf123, buf126, 4096, 128, grid=grid(4096), stream=stream0)
        buf127 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf126, buf127, 1024, 4, grid=grid(1024), stream=stream0)
        buf128 = reinterpret_tensor(buf123, (512, 1024), (1024, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (512, 1024), (1024, 1), 0), permute_360, out=buf128)
        del permute_360
        buf129 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (1, 1024), 0), view_462, out=buf129)
        buf130 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf122, buf130, 4096, 128, grid=grid(4096), stream=stream0)
        buf131 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf130, buf131, 1024, 4, grid=grid(1024), stream=stream0)
        buf132 = reinterpret_tensor(buf122, (512, 1024), (1024, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (512, 1024), (1024, 1), 0), permute_364, out=buf132)
        del permute_364
        buf133 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (1024, 512), (1, 1024), 0), view_462, out=buf133)
        del view_462
        buf134 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf121, buf134, 4096, 128, grid=grid(4096), stream=stream0)
        buf135 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf134, buf135, 1024, 4, grid=grid(1024), stream=stream0)
        buf140 = buf114; del buf114  # reuse
        buf141 = reinterpret_tensor(buf121, (1, 512, 1024), (524288, 1024, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf140, buf124, buf128, buf132, primals_340, mul_148, div_63, getitem_211, buf141, 512, 1024, grid=grid(512), stream=stream0)
        del div_63
        del getitem_211
        del primals_340
        buf138 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf139 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf124, buf128, buf132, mul_148, buf138, buf139, 1024, 512, grid=grid(1024), stream=stream0)
        del buf124
        del buf128
        del mul_148
        buf142 = reinterpret_tensor(buf105, (512, 4096), (4096, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (512, 1024), (1024, 1), 0), permute_368, out=buf142)
        del permute_368
        buf143 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (1024, 512), (1, 1024), 0), view_460, out=buf143)
        del view_460
        buf144 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf141, buf144, 4096, 128, grid=grid(4096), stream=stream0)
        buf145 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf144, buf145, 1024, 4, grid=grid(1024), stream=stream0)
        buf146 = reinterpret_tensor(buf142, (1, 512, 4096), (2097152, 4096, 1), 0); del buf142  # reuse
        # Source Nodes: [intermediate_output_20], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf146, addmm_124, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_124
        buf147 = reinterpret_tensor(buf141, (512, 1024), (1024, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (512, 4096), (4096, 1), 0), permute_372, out=buf147)
        del permute_372
        buf148 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (4096, 512), (1, 4096), 0), view_458, out=buf148)
        del view_458
        buf149 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf146, buf149, 16384, 128, grid=grid(16384), stream=stream0)
        buf150 = reinterpret_tensor(buf144, (1, 4096), (4096, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf149, buf150, 4096, 4, grid=grid(4096), stream=stream0)
        buf155 = buf140; del buf140  # reuse
        buf156 = reinterpret_tensor(buf132, (1, 512, 1024), (524288, 1024, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf155, buf147, primals_334, mul_143, div_64, getitem_207, buf156, 512, 1024, grid=grid(512), stream=stream0)
        del div_64
        del getitem_207
        del primals_334
        buf153 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf154 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf147, mul_143, buf153, buf154, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_143
        buf157 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (512, 1024), (1024, 1), 0), permute_376, out=buf157)
        del permute_376
        buf158 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (1024, 512), (1, 1024), 0), view_456, out=buf158)
        del view_456
        buf159 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf156, buf159, 4096, 128, grid=grid(4096), stream=stream0)
        del buf156
        buf160 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf159, buf160, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf161 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf157, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_268, getitem_269, getitem_270, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_7
        del clone_default_10
        del clone_default_11
        del clone_default_9
        del getitem_268
        del getitem_269
        del getitem_270
        buf162 = buf161[0]
        buf163 = buf161[1]
        buf164 = buf161[2]
        del buf161
        buf165 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (512, 1024), (1024, 1), 0), permute_388, out=buf165)
        del permute_388
        buf166 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (1024, 512), (1, 1024), 0), view_440, out=buf166)
        buf167 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf164, buf167, 4096, 128, grid=grid(4096), stream=stream0)
        buf168 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf167, buf168, 1024, 4, grid=grid(1024), stream=stream0)
        buf169 = reinterpret_tensor(buf164, (512, 1024), (1024, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (512, 1024), (1024, 1), 0), permute_393, out=buf169)
        del permute_393
        buf170 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (1024, 512), (1, 1024), 0), view_440, out=buf170)
        buf171 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf163, buf171, 4096, 128, grid=grid(4096), stream=stream0)
        buf172 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf171, buf172, 1024, 4, grid=grid(1024), stream=stream0)
        buf173 = reinterpret_tensor(buf163, (512, 1024), (1024, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (512, 1024), (1024, 1), 0), permute_397, out=buf173)
        del permute_397
        buf174 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (1024, 512), (1, 1024), 0), view_440, out=buf174)
        del view_440
        buf175 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf162, buf175, 4096, 128, grid=grid(4096), stream=stream0)
        buf176 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf175, buf176, 1024, 4, grid=grid(1024), stream=stream0)
        buf181 = buf155; del buf155  # reuse
        buf182 = reinterpret_tensor(buf162, (1, 512, 1024), (524288, 1024, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf181, buf165, buf169, buf173, primals_324, mul_141, div_66, getitem_201, buf182, 512, 1024, grid=grid(512), stream=stream0)
        del div_66
        del getitem_201
        del primals_324
        buf179 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf180 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf165, buf169, buf173, mul_141, buf179, buf180, 1024, 512, grid=grid(1024), stream=stream0)
        del buf165
        del buf169
        del mul_141
        buf183 = reinterpret_tensor(buf146, (512, 4096), (4096, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (512, 1024), (1024, 1), 0), permute_401, out=buf183)
        del permute_401
        buf184 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (1024, 512), (1, 1024), 0), view_438, out=buf184)
        del view_438
        buf185 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf182, buf185, 4096, 128, grid=grid(4096), stream=stream0)
        buf186 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf185, buf186, 1024, 4, grid=grid(1024), stream=stream0)
        buf187 = reinterpret_tensor(buf183, (1, 512, 4096), (2097152, 4096, 1), 0); del buf183  # reuse
        # Source Nodes: [intermediate_output_19], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf187, addmm_118, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_118
        buf188 = reinterpret_tensor(buf182, (512, 1024), (1024, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (512, 4096), (4096, 1), 0), permute_405, out=buf188)
        del permute_405
        buf189 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (4096, 512), (1, 4096), 0), view_436, out=buf189)
        del view_436
        buf190 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf187, buf190, 16384, 128, grid=grid(16384), stream=stream0)
        buf191 = reinterpret_tensor(buf185, (1, 4096), (4096, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf190, buf191, 4096, 4, grid=grid(4096), stream=stream0)
        buf196 = buf181; del buf181  # reuse
        buf197 = reinterpret_tensor(buf173, (1, 512, 1024), (524288, 1024, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf196, buf188, primals_318, mul_136, div_67, getitem_197, buf197, 512, 1024, grid=grid(512), stream=stream0)
        del div_67
        del getitem_197
        del primals_318
        buf194 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf195 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf188, mul_136, buf194, buf195, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_136
        buf198 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (512, 1024), (1024, 1), 0), permute_409, out=buf198)
        del permute_409
        buf199 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (1024, 512), (1, 1024), 0), view_434, out=buf199)
        del view_434
        buf200 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf197, buf200, 4096, 128, grid=grid(4096), stream=stream0)
        del buf197
        buf201 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf200, buf201, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf202 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf198, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_275, getitem_276, getitem_277, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_9
        del clone_default_12
        del clone_default_13
        del clone_default_14
        del getitem_275
        del getitem_276
        del getitem_277
        buf203 = buf202[0]
        buf204 = buf202[1]
        buf205 = buf202[2]
        del buf202
        buf206 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (512, 1024), (1024, 1), 0), permute_421, out=buf206)
        del permute_421
        buf207 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (1024, 512), (1, 1024), 0), view_418, out=buf207)
        buf208 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf205, buf208, 4096, 128, grid=grid(4096), stream=stream0)
        buf209 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf208, buf209, 1024, 4, grid=grid(1024), stream=stream0)
        buf210 = reinterpret_tensor(buf205, (512, 1024), (1024, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (512, 1024), (1024, 1), 0), permute_426, out=buf210)
        del permute_426
        buf211 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (1024, 512), (1, 1024), 0), view_418, out=buf211)
        buf212 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf204, buf212, 4096, 128, grid=grid(4096), stream=stream0)
        buf213 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf212, buf213, 1024, 4, grid=grid(1024), stream=stream0)
        buf214 = reinterpret_tensor(buf204, (512, 1024), (1024, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (512, 1024), (1024, 1), 0), permute_430, out=buf214)
        del permute_430
        buf215 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (1024, 512), (1, 1024), 0), view_418, out=buf215)
        del view_418
        buf216 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf203, buf216, 4096, 128, grid=grid(4096), stream=stream0)
        buf217 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf216, buf217, 1024, 4, grid=grid(1024), stream=stream0)
        buf222 = buf196; del buf196  # reuse
        buf223 = reinterpret_tensor(buf203, (1, 512, 1024), (524288, 1024, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf222, buf206, buf210, buf214, primals_308, mul_134, div_69, getitem_191, buf223, 512, 1024, grid=grid(512), stream=stream0)
        del div_69
        del getitem_191
        del primals_308
        buf220 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf221 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf206, buf210, buf214, mul_134, buf220, buf221, 1024, 512, grid=grid(1024), stream=stream0)
        del buf206
        del buf210
        del mul_134
        buf224 = reinterpret_tensor(buf187, (512, 4096), (4096, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (512, 1024), (1024, 1), 0), permute_434, out=buf224)
        del permute_434
        buf225 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (1024, 512), (1, 1024), 0), view_416, out=buf225)
        del view_416
        buf226 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf223, buf226, 4096, 128, grid=grid(4096), stream=stream0)
        buf227 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf226, buf227, 1024, 4, grid=grid(1024), stream=stream0)
        buf228 = reinterpret_tensor(buf224, (1, 512, 4096), (2097152, 4096, 1), 0); del buf224  # reuse
        # Source Nodes: [intermediate_output_18], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf228, addmm_112, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_112
        buf229 = reinterpret_tensor(buf223, (512, 1024), (1024, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (512, 4096), (4096, 1), 0), permute_438, out=buf229)
        del permute_438
        buf230 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (4096, 512), (1, 4096), 0), view_414, out=buf230)
        del view_414
        buf231 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf228, buf231, 16384, 128, grid=grid(16384), stream=stream0)
        buf232 = reinterpret_tensor(buf226, (1, 4096), (4096, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf231, buf232, 4096, 4, grid=grid(4096), stream=stream0)
        buf237 = buf222; del buf222  # reuse
        buf238 = reinterpret_tensor(buf214, (1, 512, 1024), (524288, 1024, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf237, buf229, primals_302, mul_129, div_70, getitem_187, buf238, 512, 1024, grid=grid(512), stream=stream0)
        del div_70
        del getitem_187
        del primals_302
        buf235 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf236 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf229, mul_129, buf235, buf236, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_129
        buf239 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (512, 1024), (1024, 1), 0), permute_442, out=buf239)
        del permute_442
        buf240 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (1024, 512), (1, 1024), 0), view_412, out=buf240)
        del view_412
        buf241 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf238, buf241, 4096, 128, grid=grid(4096), stream=stream0)
        del buf238
        buf242 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf241, buf242, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf243 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf239, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_282, getitem_283, getitem_284, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_11
        del clone_default_15
        del clone_default_16
        del clone_default_17
        del getitem_282
        del getitem_283
        del getitem_284
        buf244 = buf243[0]
        buf245 = buf243[1]
        buf246 = buf243[2]
        del buf243
        buf247 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (512, 1024), (1024, 1), 0), permute_454, out=buf247)
        del permute_454
        buf248 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (1024, 512), (1, 1024), 0), view_396, out=buf248)
        buf249 = buf241; del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf246, buf249, 4096, 128, grid=grid(4096), stream=stream0)
        buf250 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf249, buf250, 1024, 4, grid=grid(1024), stream=stream0)
        buf251 = reinterpret_tensor(buf246, (512, 1024), (1024, 1), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (512, 1024), (1024, 1), 0), permute_459, out=buf251)
        del permute_459
        buf252 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (1024, 512), (1, 1024), 0), view_396, out=buf252)
        buf253 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf245, buf253, 4096, 128, grid=grid(4096), stream=stream0)
        buf254 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf253, buf254, 1024, 4, grid=grid(1024), stream=stream0)
        buf255 = reinterpret_tensor(buf245, (512, 1024), (1024, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (512, 1024), (1024, 1), 0), permute_463, out=buf255)
        del permute_463
        buf256 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (1024, 512), (1, 1024), 0), view_396, out=buf256)
        del view_396
        buf257 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf244, buf257, 4096, 128, grid=grid(4096), stream=stream0)
        buf258 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf257, buf258, 1024, 4, grid=grid(1024), stream=stream0)
        buf263 = buf237; del buf237  # reuse
        buf264 = reinterpret_tensor(buf244, (1, 512, 1024), (524288, 1024, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf263, buf247, buf251, buf255, primals_292, mul_127, div_72, getitem_181, buf264, 512, 1024, grid=grid(512), stream=stream0)
        del div_72
        del getitem_181
        del primals_292
        buf261 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf262 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf247, buf251, buf255, mul_127, buf261, buf262, 1024, 512, grid=grid(1024), stream=stream0)
        del buf247
        del buf251
        del mul_127
        buf265 = reinterpret_tensor(buf228, (512, 4096), (4096, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (512, 1024), (1024, 1), 0), permute_467, out=buf265)
        del permute_467
        buf266 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (1024, 512), (1, 1024), 0), view_394, out=buf266)
        del view_394
        buf267 = buf257; del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf264, buf267, 4096, 128, grid=grid(4096), stream=stream0)
        buf268 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf267, buf268, 1024, 4, grid=grid(1024), stream=stream0)
        buf269 = reinterpret_tensor(buf265, (1, 512, 4096), (2097152, 4096, 1), 0); del buf265  # reuse
        # Source Nodes: [intermediate_output_17], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf269, addmm_106, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_106
        buf270 = reinterpret_tensor(buf264, (512, 1024), (1024, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (512, 4096), (4096, 1), 0), permute_471, out=buf270)
        del permute_471
        buf271 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (4096, 512), (1, 4096), 0), view_392, out=buf271)
        del view_392
        buf272 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf269, buf272, 16384, 128, grid=grid(16384), stream=stream0)
        buf273 = reinterpret_tensor(buf267, (1, 4096), (4096, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf272, buf273, 4096, 4, grid=grid(4096), stream=stream0)
        buf278 = buf263; del buf263  # reuse
        buf279 = reinterpret_tensor(buf255, (1, 512, 1024), (524288, 1024, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf278, buf270, primals_286, mul_122, div_73, getitem_177, buf279, 512, 1024, grid=grid(512), stream=stream0)
        del div_73
        del getitem_177
        del primals_286
        buf276 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf277 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf270, mul_122, buf276, buf277, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_122
        buf280 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (512, 1024), (1024, 1), 0), permute_475, out=buf280)
        del permute_475
        buf281 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (1024, 512), (1, 1024), 0), view_390, out=buf281)
        del view_390
        buf282 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf279, buf282, 4096, 128, grid=grid(4096), stream=stream0)
        del buf279
        buf283 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf282, buf283, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf284 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf280, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_289, getitem_290, getitem_291, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_13
        del clone_default_18
        del clone_default_19
        del clone_default_20
        del getitem_289
        del getitem_290
        del getitem_291
        buf285 = buf284[0]
        buf286 = buf284[1]
        buf287 = buf284[2]
        del buf284
        buf288 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (512, 1024), (1024, 1), 0), permute_487, out=buf288)
        del permute_487
        buf289 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (1024, 512), (1, 1024), 0), view_374, out=buf289)
        buf290 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf287, buf290, 4096, 128, grid=grid(4096), stream=stream0)
        buf291 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf290, buf291, 1024, 4, grid=grid(1024), stream=stream0)
        buf292 = reinterpret_tensor(buf287, (512, 1024), (1024, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (512, 1024), (1024, 1), 0), permute_492, out=buf292)
        del permute_492
        buf293 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (1024, 512), (1, 1024), 0), view_374, out=buf293)
        buf294 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf286, buf294, 4096, 128, grid=grid(4096), stream=stream0)
        buf295 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf294, buf295, 1024, 4, grid=grid(1024), stream=stream0)
        buf296 = reinterpret_tensor(buf286, (512, 1024), (1024, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (512, 1024), (1024, 1), 0), permute_496, out=buf296)
        del permute_496
        buf297 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (1024, 512), (1, 1024), 0), view_374, out=buf297)
        del view_374
        buf298 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf285, buf298, 4096, 128, grid=grid(4096), stream=stream0)
        buf299 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf298, buf299, 1024, 4, grid=grid(1024), stream=stream0)
        buf304 = buf278; del buf278  # reuse
        buf305 = reinterpret_tensor(buf285, (1, 512, 1024), (524288, 1024, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf304, buf288, buf292, buf296, primals_276, mul_120, div_75, getitem_171, buf305, 512, 1024, grid=grid(512), stream=stream0)
        del div_75
        del getitem_171
        del primals_276
        buf302 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf303 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf288, buf292, buf296, mul_120, buf302, buf303, 1024, 512, grid=grid(1024), stream=stream0)
        del buf288
        del buf292
        del mul_120
        buf306 = reinterpret_tensor(buf269, (512, 4096), (4096, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (512, 1024), (1024, 1), 0), permute_500, out=buf306)
        del permute_500
        buf307 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (1024, 512), (1, 1024), 0), view_372, out=buf307)
        del view_372
        buf308 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf305, buf308, 4096, 128, grid=grid(4096), stream=stream0)
        buf309 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf308, buf309, 1024, 4, grid=grid(1024), stream=stream0)
        buf310 = reinterpret_tensor(buf306, (1, 512, 4096), (2097152, 4096, 1), 0); del buf306  # reuse
        # Source Nodes: [intermediate_output_16], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf310, addmm_100, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_100
        buf311 = reinterpret_tensor(buf305, (512, 1024), (1024, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (512, 4096), (4096, 1), 0), permute_504, out=buf311)
        del permute_504
        buf312 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (4096, 512), (1, 4096), 0), view_370, out=buf312)
        del view_370
        buf313 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf310, buf313, 16384, 128, grid=grid(16384), stream=stream0)
        buf314 = reinterpret_tensor(buf308, (1, 4096), (4096, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf313, buf314, 4096, 4, grid=grid(4096), stream=stream0)
        buf319 = buf304; del buf304  # reuse
        buf320 = reinterpret_tensor(buf296, (1, 512, 1024), (524288, 1024, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf319, buf311, primals_270, mul_115, div_76, getitem_167, buf320, 512, 1024, grid=grid(512), stream=stream0)
        del div_76
        del getitem_167
        del primals_270
        buf317 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf318 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf311, mul_115, buf317, buf318, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_115
        buf321 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (512, 1024), (1024, 1), 0), permute_508, out=buf321)
        del permute_508
        buf322 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (1024, 512), (1, 1024), 0), view_368, out=buf322)
        del view_368
        buf323 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf320, buf323, 4096, 128, grid=grid(4096), stream=stream0)
        del buf320
        buf324 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf323, buf324, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf325 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf321, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_296, getitem_297, getitem_298, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_15
        del clone_default_21
        del clone_default_22
        del clone_default_23
        del getitem_296
        del getitem_297
        del getitem_298
        buf326 = buf325[0]
        buf327 = buf325[1]
        buf328 = buf325[2]
        del buf325
        buf329 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf328, (512, 1024), (1024, 1), 0), permute_520, out=buf329)
        del permute_520
        buf330 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf328, (1024, 512), (1, 1024), 0), view_352, out=buf330)
        buf331 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf328, buf331, 4096, 128, grid=grid(4096), stream=stream0)
        buf332 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf331, buf332, 1024, 4, grid=grid(1024), stream=stream0)
        buf333 = reinterpret_tensor(buf328, (512, 1024), (1024, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (512, 1024), (1024, 1), 0), permute_525, out=buf333)
        del permute_525
        buf334 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (1024, 512), (1, 1024), 0), view_352, out=buf334)
        buf335 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf327, buf335, 4096, 128, grid=grid(4096), stream=stream0)
        buf336 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf335, buf336, 1024, 4, grid=grid(1024), stream=stream0)
        buf337 = reinterpret_tensor(buf327, (512, 1024), (1024, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (512, 1024), (1024, 1), 0), permute_529, out=buf337)
        del permute_529
        buf338 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (1024, 512), (1, 1024), 0), view_352, out=buf338)
        del view_352
        buf339 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf326, buf339, 4096, 128, grid=grid(4096), stream=stream0)
        buf340 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf339, buf340, 1024, 4, grid=grid(1024), stream=stream0)
        buf345 = buf319; del buf319  # reuse
        buf346 = reinterpret_tensor(buf326, (1, 512, 1024), (524288, 1024, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf345, buf329, buf333, buf337, primals_260, mul_113, div_78, getitem_161, buf346, 512, 1024, grid=grid(512), stream=stream0)
        del div_78
        del getitem_161
        del primals_260
        buf343 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf344 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf329, buf333, buf337, mul_113, buf343, buf344, 1024, 512, grid=grid(1024), stream=stream0)
        del buf329
        del buf333
        del mul_113
        buf347 = reinterpret_tensor(buf310, (512, 4096), (4096, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (512, 1024), (1024, 1), 0), permute_533, out=buf347)
        del permute_533
        buf348 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (1024, 512), (1, 1024), 0), view_350, out=buf348)
        del view_350
        buf349 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf346, buf349, 4096, 128, grid=grid(4096), stream=stream0)
        buf350 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf349, buf350, 1024, 4, grid=grid(1024), stream=stream0)
        buf351 = reinterpret_tensor(buf347, (1, 512, 4096), (2097152, 4096, 1), 0); del buf347  # reuse
        # Source Nodes: [intermediate_output_15], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf351, addmm_94, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_94
        buf352 = reinterpret_tensor(buf346, (512, 1024), (1024, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf351, (512, 4096), (4096, 1), 0), permute_537, out=buf352)
        del permute_537
        buf353 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf351, (4096, 512), (1, 4096), 0), view_348, out=buf353)
        del view_348
        buf354 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf351, buf354, 16384, 128, grid=grid(16384), stream=stream0)
        buf355 = reinterpret_tensor(buf349, (1, 4096), (4096, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf354, buf355, 4096, 4, grid=grid(4096), stream=stream0)
        buf360 = buf345; del buf345  # reuse
        buf361 = reinterpret_tensor(buf337, (1, 512, 1024), (524288, 1024, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf360, buf352, primals_254, mul_108, div_79, getitem_157, buf361, 512, 1024, grid=grid(512), stream=stream0)
        del div_79
        del getitem_157
        del primals_254
        buf358 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf359 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf352, mul_108, buf358, buf359, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_108
        buf362 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf361, (512, 1024), (1024, 1), 0), permute_541, out=buf362)
        del permute_541
        buf363 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf361, (1024, 512), (1, 1024), 0), view_346, out=buf363)
        del view_346
        buf364 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf361, buf364, 4096, 128, grid=grid(4096), stream=stream0)
        del buf361
        buf365 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf364, buf365, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf366 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf362, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_303, getitem_304, getitem_305, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_17
        del clone_default_24
        del clone_default_25
        del clone_default_26
        del getitem_303
        del getitem_304
        del getitem_305
        buf367 = buf366[0]
        buf368 = buf366[1]
        buf369 = buf366[2]
        del buf366
        buf370 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (512, 1024), (1024, 1), 0), permute_553, out=buf370)
        del permute_553
        buf371 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (1024, 512), (1, 1024), 0), view_330, out=buf371)
        buf372 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf369, buf372, 4096, 128, grid=grid(4096), stream=stream0)
        buf373 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf372, buf373, 1024, 4, grid=grid(1024), stream=stream0)
        buf374 = reinterpret_tensor(buf369, (512, 1024), (1024, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (512, 1024), (1024, 1), 0), permute_558, out=buf374)
        del permute_558
        buf375 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (1024, 512), (1, 1024), 0), view_330, out=buf375)
        buf376 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf368, buf376, 4096, 128, grid=grid(4096), stream=stream0)
        buf377 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf376, buf377, 1024, 4, grid=grid(1024), stream=stream0)
        buf378 = reinterpret_tensor(buf368, (512, 1024), (1024, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (512, 1024), (1024, 1), 0), permute_562, out=buf378)
        del permute_562
        buf379 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (1024, 512), (1, 1024), 0), view_330, out=buf379)
        del view_330
        buf380 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf367, buf380, 4096, 128, grid=grid(4096), stream=stream0)
        buf381 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf380, buf381, 1024, 4, grid=grid(1024), stream=stream0)
        buf386 = buf360; del buf360  # reuse
        buf387 = reinterpret_tensor(buf367, (1, 512, 1024), (524288, 1024, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf386, buf370, buf374, buf378, primals_244, mul_106, div_81, getitem_151, buf387, 512, 1024, grid=grid(512), stream=stream0)
        del div_81
        del getitem_151
        del primals_244
        buf384 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf385 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf370, buf374, buf378, mul_106, buf384, buf385, 1024, 512, grid=grid(1024), stream=stream0)
        del buf370
        del buf374
        del mul_106
        buf388 = reinterpret_tensor(buf351, (512, 4096), (4096, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (512, 1024), (1024, 1), 0), permute_566, out=buf388)
        del permute_566
        buf389 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (1024, 512), (1, 1024), 0), view_328, out=buf389)
        del view_328
        buf390 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf387, buf390, 4096, 128, grid=grid(4096), stream=stream0)
        buf391 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf390, buf391, 1024, 4, grid=grid(1024), stream=stream0)
        buf392 = reinterpret_tensor(buf388, (1, 512, 4096), (2097152, 4096, 1), 0); del buf388  # reuse
        # Source Nodes: [intermediate_output_14], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf392, addmm_88, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_88
        buf393 = reinterpret_tensor(buf387, (512, 1024), (1024, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (512, 4096), (4096, 1), 0), permute_570, out=buf393)
        del permute_570
        buf394 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (4096, 512), (1, 4096), 0), view_326, out=buf394)
        del view_326
        buf395 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf392, buf395, 16384, 128, grid=grid(16384), stream=stream0)
        buf396 = reinterpret_tensor(buf390, (1, 4096), (4096, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf395, buf396, 4096, 4, grid=grid(4096), stream=stream0)
        buf401 = buf386; del buf386  # reuse
        buf402 = reinterpret_tensor(buf378, (1, 512, 1024), (524288, 1024, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf401, buf393, primals_238, mul_101, div_82, getitem_147, buf402, 512, 1024, grid=grid(512), stream=stream0)
        del div_82
        del getitem_147
        del primals_238
        buf399 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf400 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf393, mul_101, buf399, buf400, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_101
        buf403 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf402, (512, 1024), (1024, 1), 0), permute_574, out=buf403)
        del permute_574
        buf404 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf402, (1024, 512), (1, 1024), 0), view_324, out=buf404)
        del view_324
        buf405 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf402, buf405, 4096, 128, grid=grid(4096), stream=stream0)
        del buf402
        buf406 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf405, buf406, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf407 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf403, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_310, getitem_311, getitem_312, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_19
        del clone_default_27
        del clone_default_28
        del clone_default_29
        del getitem_310
        del getitem_311
        del getitem_312
        buf408 = buf407[0]
        buf409 = buf407[1]
        buf410 = buf407[2]
        del buf407
        buf411 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (512, 1024), (1024, 1), 0), permute_586, out=buf411)
        del permute_586
        buf412 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (1024, 512), (1, 1024), 0), view_308, out=buf412)
        buf413 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf410, buf413, 4096, 128, grid=grid(4096), stream=stream0)
        buf414 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf413, buf414, 1024, 4, grid=grid(1024), stream=stream0)
        buf415 = reinterpret_tensor(buf410, (512, 1024), (1024, 1), 0); del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (512, 1024), (1024, 1), 0), permute_591, out=buf415)
        del permute_591
        buf416 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (1024, 512), (1, 1024), 0), view_308, out=buf416)
        buf417 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf409, buf417, 4096, 128, grid=grid(4096), stream=stream0)
        buf418 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf417, buf418, 1024, 4, grid=grid(1024), stream=stream0)
        buf419 = reinterpret_tensor(buf409, (512, 1024), (1024, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf408, (512, 1024), (1024, 1), 0), permute_595, out=buf419)
        del permute_595
        buf420 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf408, (1024, 512), (1, 1024), 0), view_308, out=buf420)
        del view_308
        buf421 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf408, buf421, 4096, 128, grid=grid(4096), stream=stream0)
        buf422 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf421, buf422, 1024, 4, grid=grid(1024), stream=stream0)
        buf427 = buf401; del buf401  # reuse
        buf428 = reinterpret_tensor(buf408, (1, 512, 1024), (524288, 1024, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf427, buf411, buf415, buf419, primals_228, mul_99, div_84, getitem_141, buf428, 512, 1024, grid=grid(512), stream=stream0)
        del div_84
        del getitem_141
        del primals_228
        buf425 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf426 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf411, buf415, buf419, mul_99, buf425, buf426, 1024, 512, grid=grid(1024), stream=stream0)
        del buf411
        del buf415
        del mul_99
        buf429 = reinterpret_tensor(buf392, (512, 4096), (4096, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (512, 1024), (1024, 1), 0), permute_599, out=buf429)
        del permute_599
        buf430 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (1024, 512), (1, 1024), 0), view_306, out=buf430)
        del view_306
        buf431 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf428, buf431, 4096, 128, grid=grid(4096), stream=stream0)
        buf432 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf431, buf432, 1024, 4, grid=grid(1024), stream=stream0)
        buf433 = reinterpret_tensor(buf429, (1, 512, 4096), (2097152, 4096, 1), 0); del buf429  # reuse
        # Source Nodes: [intermediate_output_13], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf433, addmm_82, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_82
        buf434 = reinterpret_tensor(buf428, (512, 1024), (1024, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (512, 4096), (4096, 1), 0), permute_603, out=buf434)
        del permute_603
        buf435 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (4096, 512), (1, 4096), 0), view_304, out=buf435)
        del view_304
        buf436 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf433, buf436, 16384, 128, grid=grid(16384), stream=stream0)
        buf437 = reinterpret_tensor(buf431, (1, 4096), (4096, 1), 0); del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf436, buf437, 4096, 4, grid=grid(4096), stream=stream0)
        buf442 = buf427; del buf427  # reuse
        buf443 = reinterpret_tensor(buf419, (1, 512, 1024), (524288, 1024, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf442, buf434, primals_222, mul_94, div_85, getitem_137, buf443, 512, 1024, grid=grid(512), stream=stream0)
        del div_85
        del getitem_137
        del primals_222
        buf440 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf441 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf434, mul_94, buf440, buf441, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_94
        buf444 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (512, 1024), (1024, 1), 0), permute_607, out=buf444)
        del permute_607
        buf445 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (1024, 512), (1, 1024), 0), view_302, out=buf445)
        del view_302
        buf446 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf443, buf446, 4096, 128, grid=grid(4096), stream=stream0)
        del buf443
        buf447 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf446, buf447, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf448 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf444, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_317, getitem_318, getitem_319, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_21
        del clone_default_30
        del clone_default_31
        del clone_default_32
        del getitem_317
        del getitem_318
        del getitem_319
        buf449 = buf448[0]
        buf450 = buf448[1]
        buf451 = buf448[2]
        del buf448
        buf452 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf451, (512, 1024), (1024, 1), 0), permute_619, out=buf452)
        del permute_619
        buf453 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf451, (1024, 512), (1, 1024), 0), view_286, out=buf453)
        buf454 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf451, buf454, 4096, 128, grid=grid(4096), stream=stream0)
        buf455 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf454, buf455, 1024, 4, grid=grid(1024), stream=stream0)
        buf456 = reinterpret_tensor(buf451, (512, 1024), (1024, 1), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (512, 1024), (1024, 1), 0), permute_624, out=buf456)
        del permute_624
        buf457 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (1024, 512), (1, 1024), 0), view_286, out=buf457)
        buf458 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf450, buf458, 4096, 128, grid=grid(4096), stream=stream0)
        buf459 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf458, buf459, 1024, 4, grid=grid(1024), stream=stream0)
        buf460 = reinterpret_tensor(buf450, (512, 1024), (1024, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (512, 1024), (1024, 1), 0), permute_628, out=buf460)
        del permute_628
        buf461 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (1024, 512), (1, 1024), 0), view_286, out=buf461)
        del view_286
        buf462 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf449, buf462, 4096, 128, grid=grid(4096), stream=stream0)
        buf463 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf462, buf463, 1024, 4, grid=grid(1024), stream=stream0)
        buf468 = buf442; del buf442  # reuse
        buf469 = reinterpret_tensor(buf449, (1, 512, 1024), (524288, 1024, 1), 0); del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf468, buf452, buf456, buf460, primals_212, mul_92, div_87, getitem_131, buf469, 512, 1024, grid=grid(512), stream=stream0)
        del div_87
        del getitem_131
        del primals_212
        buf466 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf467 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf452, buf456, buf460, mul_92, buf466, buf467, 1024, 512, grid=grid(1024), stream=stream0)
        del buf452
        del buf456
        del mul_92
        buf470 = reinterpret_tensor(buf433, (512, 4096), (4096, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (512, 1024), (1024, 1), 0), permute_632, out=buf470)
        del permute_632
        buf471 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (1024, 512), (1, 1024), 0), view_284, out=buf471)
        del view_284
        buf472 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf469, buf472, 4096, 128, grid=grid(4096), stream=stream0)
        buf473 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf472, buf473, 1024, 4, grid=grid(1024), stream=stream0)
        buf474 = reinterpret_tensor(buf470, (1, 512, 4096), (2097152, 4096, 1), 0); del buf470  # reuse
        # Source Nodes: [intermediate_output_12], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf474, addmm_76, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_76
        buf475 = reinterpret_tensor(buf469, (512, 1024), (1024, 1), 0); del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (512, 4096), (4096, 1), 0), permute_636, out=buf475)
        del permute_636
        buf476 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (4096, 512), (1, 4096), 0), view_282, out=buf476)
        del view_282
        buf477 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf474, buf477, 16384, 128, grid=grid(16384), stream=stream0)
        buf478 = reinterpret_tensor(buf472, (1, 4096), (4096, 1), 0); del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf477, buf478, 4096, 4, grid=grid(4096), stream=stream0)
        buf483 = buf468; del buf468  # reuse
        buf484 = reinterpret_tensor(buf460, (1, 512, 1024), (524288, 1024, 1), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf483, buf475, primals_206, mul_87, div_88, getitem_127, buf484, 512, 1024, grid=grid(512), stream=stream0)
        del div_88
        del getitem_127
        del primals_206
        buf481 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf482 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf475, mul_87, buf481, buf482, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_87
        buf485 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (512, 1024), (1024, 1), 0), permute_640, out=buf485)
        del permute_640
        buf486 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (1024, 512), (1, 1024), 0), view_280, out=buf486)
        del view_280
        buf487 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf484, buf487, 4096, 128, grid=grid(4096), stream=stream0)
        del buf484
        buf488 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf487, buf488, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf489 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf485, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_324, getitem_325, getitem_326, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_23
        del clone_default_33
        del clone_default_34
        del clone_default_35
        del getitem_324
        del getitem_325
        del getitem_326
        buf490 = buf489[0]
        buf491 = buf489[1]
        buf492 = buf489[2]
        del buf489
        buf493 = buf485; del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf492, (512, 1024), (1024, 1), 0), permute_652, out=buf493)
        del permute_652
        buf494 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf492, (1024, 512), (1, 1024), 0), view_264, out=buf494)
        buf495 = buf487; del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf492, buf495, 4096, 128, grid=grid(4096), stream=stream0)
        buf496 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf495, buf496, 1024, 4, grid=grid(1024), stream=stream0)
        buf497 = reinterpret_tensor(buf492, (512, 1024), (1024, 1), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf491, (512, 1024), (1024, 1), 0), permute_657, out=buf497)
        del permute_657
        buf498 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf491, (1024, 512), (1, 1024), 0), view_264, out=buf498)
        buf499 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf491, buf499, 4096, 128, grid=grid(4096), stream=stream0)
        buf500 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf499, buf500, 1024, 4, grid=grid(1024), stream=stream0)
        buf501 = reinterpret_tensor(buf491, (512, 1024), (1024, 1), 0); del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (512, 1024), (1024, 1), 0), permute_661, out=buf501)
        del permute_661
        buf502 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (1024, 512), (1, 1024), 0), view_264, out=buf502)
        del view_264
        buf503 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf490, buf503, 4096, 128, grid=grid(4096), stream=stream0)
        buf504 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf503, buf504, 1024, 4, grid=grid(1024), stream=stream0)
        buf509 = buf483; del buf483  # reuse
        buf510 = reinterpret_tensor(buf490, (1, 512, 1024), (524288, 1024, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf509, buf493, buf497, buf501, primals_196, mul_85, div_90, getitem_121, buf510, 512, 1024, grid=grid(512), stream=stream0)
        del div_90
        del getitem_121
        del primals_196
        buf507 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf508 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf493, buf497, buf501, mul_85, buf507, buf508, 1024, 512, grid=grid(1024), stream=stream0)
        del buf493
        del buf497
        del mul_85
        buf511 = reinterpret_tensor(buf474, (512, 4096), (4096, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf510, (512, 1024), (1024, 1), 0), permute_665, out=buf511)
        del permute_665
        buf512 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf510, (1024, 512), (1, 1024), 0), view_262, out=buf512)
        del view_262
        buf513 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf510, buf513, 4096, 128, grid=grid(4096), stream=stream0)
        buf514 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf513, buf514, 1024, 4, grid=grid(1024), stream=stream0)
        buf515 = reinterpret_tensor(buf511, (1, 512, 4096), (2097152, 4096, 1), 0); del buf511  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf515, addmm_70, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_70
        buf516 = reinterpret_tensor(buf510, (512, 1024), (1024, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf515, (512, 4096), (4096, 1), 0), permute_669, out=buf516)
        del permute_669
        buf517 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf515, (4096, 512), (1, 4096), 0), view_260, out=buf517)
        del view_260
        buf518 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf515, buf518, 16384, 128, grid=grid(16384), stream=stream0)
        buf519 = reinterpret_tensor(buf513, (1, 4096), (4096, 1), 0); del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf518, buf519, 4096, 4, grid=grid(4096), stream=stream0)
        buf524 = buf509; del buf509  # reuse
        buf525 = reinterpret_tensor(buf501, (1, 512, 1024), (524288, 1024, 1), 0); del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf524, buf516, primals_190, mul_80, div_91, getitem_117, buf525, 512, 1024, grid=grid(512), stream=stream0)
        del div_91
        del getitem_117
        del primals_190
        buf522 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf523 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf516, mul_80, buf522, buf523, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_80
        buf526 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (512, 1024), (1024, 1), 0), permute_673, out=buf526)
        del permute_673
        buf527 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (1024, 512), (1, 1024), 0), view_258, out=buf527)
        del view_258
        buf528 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf525, buf528, 4096, 128, grid=grid(4096), stream=stream0)
        del buf525
        buf529 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf528, buf529, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf530 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf526, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_36, clone_default_37, clone_default_38, None, alias_default_25, getitem_331, getitem_332, getitem_333, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_25
        del clone_default_36
        del clone_default_37
        del clone_default_38
        del getitem_331
        del getitem_332
        del getitem_333
        buf531 = buf530[0]
        buf532 = buf530[1]
        buf533 = buf530[2]
        del buf530
        buf534 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (512, 1024), (1024, 1), 0), permute_685, out=buf534)
        del permute_685
        buf535 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (1024, 512), (1, 1024), 0), view_242, out=buf535)
        buf536 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf533, buf536, 4096, 128, grid=grid(4096), stream=stream0)
        buf537 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf536, buf537, 1024, 4, grid=grid(1024), stream=stream0)
        buf538 = reinterpret_tensor(buf533, (512, 1024), (1024, 1), 0); del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (512, 1024), (1024, 1), 0), permute_690, out=buf538)
        del permute_690
        buf539 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (1024, 512), (1, 1024), 0), view_242, out=buf539)
        buf540 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf532, buf540, 4096, 128, grid=grid(4096), stream=stream0)
        buf541 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf540, buf541, 1024, 4, grid=grid(1024), stream=stream0)
        buf542 = reinterpret_tensor(buf532, (512, 1024), (1024, 1), 0); del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (512, 1024), (1024, 1), 0), permute_694, out=buf542)
        del permute_694
        buf543 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (1024, 512), (1, 1024), 0), view_242, out=buf543)
        del view_242
        buf544 = buf540; del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf531, buf544, 4096, 128, grid=grid(4096), stream=stream0)
        buf545 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf544, buf545, 1024, 4, grid=grid(1024), stream=stream0)
        buf550 = buf524; del buf524  # reuse
        buf551 = reinterpret_tensor(buf531, (1, 512, 1024), (524288, 1024, 1), 0); del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf550, buf534, buf538, buf542, primals_180, mul_78, div_93, getitem_111, buf551, 512, 1024, grid=grid(512), stream=stream0)
        del div_93
        del getitem_111
        del primals_180
        buf548 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf549 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf534, buf538, buf542, mul_78, buf548, buf549, 1024, 512, grid=grid(1024), stream=stream0)
        del buf534
        del buf538
        del mul_78
        buf552 = reinterpret_tensor(buf515, (512, 4096), (4096, 1), 0); del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (512, 1024), (1024, 1), 0), permute_698, out=buf552)
        del permute_698
        buf553 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (1024, 512), (1, 1024), 0), view_240, out=buf553)
        del view_240
        buf554 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf551, buf554, 4096, 128, grid=grid(4096), stream=stream0)
        buf555 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf554, buf555, 1024, 4, grid=grid(1024), stream=stream0)
        buf556 = reinterpret_tensor(buf552, (1, 512, 4096), (2097152, 4096, 1), 0); del buf552  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf556, addmm_64, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_64
        buf557 = reinterpret_tensor(buf551, (512, 1024), (1024, 1), 0); del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf556, (512, 4096), (4096, 1), 0), permute_702, out=buf557)
        del permute_702
        buf558 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf556, (4096, 512), (1, 4096), 0), view_238, out=buf558)
        del view_238
        buf559 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf556, buf559, 16384, 128, grid=grid(16384), stream=stream0)
        buf560 = reinterpret_tensor(buf554, (1, 4096), (4096, 1), 0); del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf559, buf560, 4096, 4, grid=grid(4096), stream=stream0)
        buf565 = buf550; del buf550  # reuse
        buf566 = reinterpret_tensor(buf542, (1, 512, 1024), (524288, 1024, 1), 0); del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf565, buf557, primals_174, mul_73, div_94, getitem_107, buf566, 512, 1024, grid=grid(512), stream=stream0)
        del div_94
        del getitem_107
        del primals_174
        buf563 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf564 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf557, mul_73, buf563, buf564, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_73
        buf567 = buf557; del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf566, (512, 1024), (1024, 1), 0), permute_706, out=buf567)
        del permute_706
        buf568 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf566, (1024, 512), (1, 1024), 0), view_236, out=buf568)
        del view_236
        buf569 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf566, buf569, 4096, 128, grid=grid(4096), stream=stream0)
        del buf566
        buf570 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf569, buf570, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf571 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf567, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_39, clone_default_40, clone_default_41, None, alias_default_27, getitem_338, getitem_339, getitem_340, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_27
        del clone_default_39
        del clone_default_40
        del clone_default_41
        del getitem_338
        del getitem_339
        del getitem_340
        buf572 = buf571[0]
        buf573 = buf571[1]
        buf574 = buf571[2]
        del buf571
        buf575 = buf567; del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (512, 1024), (1024, 1), 0), permute_718, out=buf575)
        del permute_718
        buf576 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (1024, 512), (1, 1024), 0), view_220, out=buf576)
        buf577 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf574, buf577, 4096, 128, grid=grid(4096), stream=stream0)
        buf578 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf577, buf578, 1024, 4, grid=grid(1024), stream=stream0)
        buf579 = reinterpret_tensor(buf574, (512, 1024), (1024, 1), 0); del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf573, (512, 1024), (1024, 1), 0), permute_723, out=buf579)
        del permute_723
        buf580 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf573, (1024, 512), (1, 1024), 0), view_220, out=buf580)
        buf581 = buf577; del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf573, buf581, 4096, 128, grid=grid(4096), stream=stream0)
        buf582 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf581, buf582, 1024, 4, grid=grid(1024), stream=stream0)
        buf583 = reinterpret_tensor(buf573, (512, 1024), (1024, 1), 0); del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (512, 1024), (1024, 1), 0), permute_727, out=buf583)
        del permute_727
        buf584 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (1024, 512), (1, 1024), 0), view_220, out=buf584)
        del view_220
        buf585 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf572, buf585, 4096, 128, grid=grid(4096), stream=stream0)
        buf586 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf585, buf586, 1024, 4, grid=grid(1024), stream=stream0)
        buf591 = buf565; del buf565  # reuse
        buf592 = reinterpret_tensor(buf572, (1, 512, 1024), (524288, 1024, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf591, buf575, buf579, buf583, primals_164, mul_71, div_96, getitem_101, buf592, 512, 1024, grid=grid(512), stream=stream0)
        del div_96
        del getitem_101
        del primals_164
        buf589 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf590 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf575, buf579, buf583, mul_71, buf589, buf590, 1024, 512, grid=grid(1024), stream=stream0)
        del buf575
        del buf579
        del mul_71
        buf593 = reinterpret_tensor(buf556, (512, 4096), (4096, 1), 0); del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf592, (512, 1024), (1024, 1), 0), permute_731, out=buf593)
        del permute_731
        buf594 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf592, (1024, 512), (1, 1024), 0), view_218, out=buf594)
        del view_218
        buf595 = buf585; del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf592, buf595, 4096, 128, grid=grid(4096), stream=stream0)
        buf596 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf595, buf596, 1024, 4, grid=grid(1024), stream=stream0)
        buf597 = reinterpret_tensor(buf593, (1, 512, 4096), (2097152, 4096, 1), 0); del buf593  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf597, addmm_58, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_58
        buf598 = reinterpret_tensor(buf592, (512, 1024), (1024, 1), 0); del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf597, (512, 4096), (4096, 1), 0), permute_735, out=buf598)
        del permute_735
        buf599 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf597, (4096, 512), (1, 4096), 0), view_216, out=buf599)
        del view_216
        buf600 = buf559; del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf597, buf600, 16384, 128, grid=grid(16384), stream=stream0)
        buf601 = reinterpret_tensor(buf595, (1, 4096), (4096, 1), 0); del buf595  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf600, buf601, 4096, 4, grid=grid(4096), stream=stream0)
        buf606 = buf591; del buf591  # reuse
        buf607 = reinterpret_tensor(buf583, (1, 512, 1024), (524288, 1024, 1), 0); del buf583  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf606, buf598, primals_158, mul_66, div_97, getitem_97, buf607, 512, 1024, grid=grid(512), stream=stream0)
        del div_97
        del getitem_97
        del primals_158
        buf604 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf605 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf598, mul_66, buf604, buf605, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_66
        buf608 = buf598; del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf607, (512, 1024), (1024, 1), 0), permute_739, out=buf608)
        del permute_739
        buf609 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf607, (1024, 512), (1, 1024), 0), view_214, out=buf609)
        del view_214
        buf610 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf607, buf610, 4096, 128, grid=grid(4096), stream=stream0)
        del buf607
        buf611 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf610, buf611, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf612 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf608, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_42, clone_default_43, clone_default_44, None, alias_default_29, getitem_345, getitem_346, getitem_347, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_29
        del clone_default_42
        del clone_default_43
        del clone_default_44
        del getitem_345
        del getitem_346
        del getitem_347
        buf613 = buf612[0]
        buf614 = buf612[1]
        buf615 = buf612[2]
        del buf612
        buf616 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf615, (512, 1024), (1024, 1), 0), permute_751, out=buf616)
        del permute_751
        buf617 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf615, (1024, 512), (1, 1024), 0), view_198, out=buf617)
        buf618 = buf610; del buf610  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf615, buf618, 4096, 128, grid=grid(4096), stream=stream0)
        buf619 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf618, buf619, 1024, 4, grid=grid(1024), stream=stream0)
        buf620 = reinterpret_tensor(buf615, (512, 1024), (1024, 1), 0); del buf615  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf614, (512, 1024), (1024, 1), 0), permute_756, out=buf620)
        del permute_756
        buf621 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf614, (1024, 512), (1, 1024), 0), view_198, out=buf621)
        buf622 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf614, buf622, 4096, 128, grid=grid(4096), stream=stream0)
        buf623 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf622, buf623, 1024, 4, grid=grid(1024), stream=stream0)
        buf624 = reinterpret_tensor(buf614, (512, 1024), (1024, 1), 0); del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf613, (512, 1024), (1024, 1), 0), permute_760, out=buf624)
        del permute_760
        buf625 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf613, (1024, 512), (1, 1024), 0), view_198, out=buf625)
        del view_198
        buf626 = buf622; del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf613, buf626, 4096, 128, grid=grid(4096), stream=stream0)
        buf627 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf626, buf627, 1024, 4, grid=grid(1024), stream=stream0)
        buf632 = buf606; del buf606  # reuse
        buf633 = reinterpret_tensor(buf613, (1, 512, 1024), (524288, 1024, 1), 0); del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf632, buf616, buf620, buf624, primals_148, mul_64, div_99, getitem_91, buf633, 512, 1024, grid=grid(512), stream=stream0)
        del div_99
        del getitem_91
        del primals_148
        buf630 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf631 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf616, buf620, buf624, mul_64, buf630, buf631, 1024, 512, grid=grid(1024), stream=stream0)
        del buf616
        del buf620
        del mul_64
        buf634 = reinterpret_tensor(buf597, (512, 4096), (4096, 1), 0); del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf633, (512, 1024), (1024, 1), 0), permute_764, out=buf634)
        del permute_764
        buf635 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf633, (1024, 512), (1, 1024), 0), view_196, out=buf635)
        del view_196
        buf636 = buf626; del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf633, buf636, 4096, 128, grid=grid(4096), stream=stream0)
        buf637 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf636, buf637, 1024, 4, grid=grid(1024), stream=stream0)
        buf638 = reinterpret_tensor(buf634, (1, 512, 4096), (2097152, 4096, 1), 0); del buf634  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf638, addmm_52, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_52
        buf639 = reinterpret_tensor(buf633, (512, 1024), (1024, 1), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf638, (512, 4096), (4096, 1), 0), permute_768, out=buf639)
        del permute_768
        buf640 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf638, (4096, 512), (1, 4096), 0), view_194, out=buf640)
        del view_194
        buf641 = buf600; del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf638, buf641, 16384, 128, grid=grid(16384), stream=stream0)
        buf642 = reinterpret_tensor(buf636, (1, 4096), (4096, 1), 0); del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf641, buf642, 4096, 4, grid=grid(4096), stream=stream0)
        buf647 = buf632; del buf632  # reuse
        buf648 = reinterpret_tensor(buf624, (1, 512, 1024), (524288, 1024, 1), 0); del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf647, buf639, primals_142, mul_59, div_100, getitem_87, buf648, 512, 1024, grid=grid(512), stream=stream0)
        del div_100
        del getitem_87
        del primals_142
        buf645 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf646 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf639, mul_59, buf645, buf646, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_59
        buf649 = buf639; del buf639  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (512, 1024), (1024, 1), 0), permute_772, out=buf649)
        del permute_772
        buf650 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (1024, 512), (1, 1024), 0), view_192, out=buf650)
        del view_192
        buf651 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf648, buf651, 4096, 128, grid=grid(4096), stream=stream0)
        del buf648
        buf652 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf651, buf652, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf653 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf649, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_45, clone_default_46, clone_default_47, None, alias_default_31, getitem_352, getitem_353, getitem_354, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_31
        del clone_default_45
        del clone_default_46
        del clone_default_47
        del getitem_352
        del getitem_353
        del getitem_354
        buf654 = buf653[0]
        buf655 = buf653[1]
        buf656 = buf653[2]
        del buf653
        buf657 = buf649; del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf656, (512, 1024), (1024, 1), 0), permute_784, out=buf657)
        del permute_784
        buf658 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf656, (1024, 512), (1, 1024), 0), view_176, out=buf658)
        buf659 = buf651; del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf656, buf659, 4096, 128, grid=grid(4096), stream=stream0)
        buf660 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf659, buf660, 1024, 4, grid=grid(1024), stream=stream0)
        buf661 = reinterpret_tensor(buf656, (512, 1024), (1024, 1), 0); del buf656  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf655, (512, 1024), (1024, 1), 0), permute_789, out=buf661)
        del permute_789
        buf662 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf655, (1024, 512), (1, 1024), 0), view_176, out=buf662)
        buf663 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf655, buf663, 4096, 128, grid=grid(4096), stream=stream0)
        buf664 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf663, buf664, 1024, 4, grid=grid(1024), stream=stream0)
        buf665 = reinterpret_tensor(buf655, (512, 1024), (1024, 1), 0); del buf655  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf654, (512, 1024), (1024, 1), 0), permute_793, out=buf665)
        del permute_793
        buf666 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf654, (1024, 512), (1, 1024), 0), view_176, out=buf666)
        del view_176
        buf667 = buf663; del buf663  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf654, buf667, 4096, 128, grid=grid(4096), stream=stream0)
        buf668 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf667, buf668, 1024, 4, grid=grid(1024), stream=stream0)
        buf673 = buf647; del buf647  # reuse
        buf674 = reinterpret_tensor(buf654, (1, 512, 1024), (524288, 1024, 1), 0); del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf673, buf657, buf661, buf665, primals_132, mul_57, div_102, getitem_81, buf674, 512, 1024, grid=grid(512), stream=stream0)
        del div_102
        del getitem_81
        del primals_132
        buf671 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf672 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf657, buf661, buf665, mul_57, buf671, buf672, 1024, 512, grid=grid(1024), stream=stream0)
        del buf657
        del buf661
        del mul_57
        buf675 = reinterpret_tensor(buf638, (512, 4096), (4096, 1), 0); del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf674, (512, 1024), (1024, 1), 0), permute_797, out=buf675)
        del permute_797
        buf676 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf674, (1024, 512), (1, 1024), 0), view_174, out=buf676)
        del view_174
        buf677 = buf667; del buf667  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf674, buf677, 4096, 128, grid=grid(4096), stream=stream0)
        buf678 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf677, buf678, 1024, 4, grid=grid(1024), stream=stream0)
        buf679 = reinterpret_tensor(buf675, (1, 512, 4096), (2097152, 4096, 1), 0); del buf675  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf679, addmm_46, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_46
        buf680 = reinterpret_tensor(buf674, (512, 1024), (1024, 1), 0); del buf674  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf679, (512, 4096), (4096, 1), 0), permute_801, out=buf680)
        del permute_801
        buf681 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf679, (4096, 512), (1, 4096), 0), view_172, out=buf681)
        del view_172
        buf682 = buf641; del buf641  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf679, buf682, 16384, 128, grid=grid(16384), stream=stream0)
        buf683 = reinterpret_tensor(buf677, (1, 4096), (4096, 1), 0); del buf677  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf682, buf683, 4096, 4, grid=grid(4096), stream=stream0)
        buf688 = buf673; del buf673  # reuse
        buf689 = reinterpret_tensor(buf665, (1, 512, 1024), (524288, 1024, 1), 0); del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf688, buf680, primals_126, mul_52, div_103, getitem_77, buf689, 512, 1024, grid=grid(512), stream=stream0)
        del div_103
        del getitem_77
        del primals_126
        buf686 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf687 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf680, mul_52, buf686, buf687, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_52
        buf690 = buf680; del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf689, (512, 1024), (1024, 1), 0), permute_805, out=buf690)
        del permute_805
        buf691 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf689, (1024, 512), (1, 1024), 0), view_170, out=buf691)
        del view_170
        buf692 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf689, buf692, 4096, 128, grid=grid(4096), stream=stream0)
        del buf689
        buf693 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf692, buf693, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf694 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf690, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_48, clone_default_49, clone_default_50, None, alias_default_33, getitem_359, getitem_360, getitem_361, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_33
        del clone_default_48
        del clone_default_49
        del clone_default_50
        del getitem_359
        del getitem_360
        del getitem_361
        buf695 = buf694[0]
        buf696 = buf694[1]
        buf697 = buf694[2]
        del buf694
        buf698 = buf690; del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf697, (512, 1024), (1024, 1), 0), permute_817, out=buf698)
        del permute_817
        buf699 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf697, (1024, 512), (1, 1024), 0), view_154, out=buf699)
        buf700 = buf692; del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf697, buf700, 4096, 128, grid=grid(4096), stream=stream0)
        buf701 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf700, buf701, 1024, 4, grid=grid(1024), stream=stream0)
        buf702 = reinterpret_tensor(buf697, (512, 1024), (1024, 1), 0); del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf696, (512, 1024), (1024, 1), 0), permute_822, out=buf702)
        del permute_822
        buf703 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf696, (1024, 512), (1, 1024), 0), view_154, out=buf703)
        buf704 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf696, buf704, 4096, 128, grid=grid(4096), stream=stream0)
        buf705 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf704, buf705, 1024, 4, grid=grid(1024), stream=stream0)
        buf706 = reinterpret_tensor(buf696, (512, 1024), (1024, 1), 0); del buf696  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (512, 1024), (1024, 1), 0), permute_826, out=buf706)
        del permute_826
        buf707 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (1024, 512), (1, 1024), 0), view_154, out=buf707)
        del view_154
        buf708 = buf704; del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf695, buf708, 4096, 128, grid=grid(4096), stream=stream0)
        buf709 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf708, buf709, 1024, 4, grid=grid(1024), stream=stream0)
        buf714 = buf688; del buf688  # reuse
        buf715 = reinterpret_tensor(buf695, (1, 512, 1024), (524288, 1024, 1), 0); del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf714, buf698, buf702, buf706, primals_116, mul_50, div_105, getitem_71, buf715, 512, 1024, grid=grid(512), stream=stream0)
        del div_105
        del getitem_71
        del primals_116
        buf712 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf713 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf698, buf702, buf706, mul_50, buf712, buf713, 1024, 512, grid=grid(1024), stream=stream0)
        del buf698
        del buf702
        del mul_50
        buf716 = reinterpret_tensor(buf679, (512, 4096), (4096, 1), 0); del buf679  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf715, (512, 1024), (1024, 1), 0), permute_830, out=buf716)
        del permute_830
        buf717 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf715, (1024, 512), (1, 1024), 0), view_152, out=buf717)
        del view_152
        buf718 = buf708; del buf708  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf715, buf718, 4096, 128, grid=grid(4096), stream=stream0)
        buf719 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf718, buf719, 1024, 4, grid=grid(1024), stream=stream0)
        buf720 = reinterpret_tensor(buf716, (1, 512, 4096), (2097152, 4096, 1), 0); del buf716  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf720, addmm_40, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_40
        buf721 = reinterpret_tensor(buf715, (512, 1024), (1024, 1), 0); del buf715  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf720, (512, 4096), (4096, 1), 0), permute_834, out=buf721)
        del permute_834
        buf722 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf720, (4096, 512), (1, 4096), 0), view_150, out=buf722)
        del view_150
        buf723 = buf682; del buf682  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf720, buf723, 16384, 128, grid=grid(16384), stream=stream0)
        buf724 = reinterpret_tensor(buf718, (1, 4096), (4096, 1), 0); del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf723, buf724, 4096, 4, grid=grid(4096), stream=stream0)
        buf729 = buf714; del buf714  # reuse
        buf730 = reinterpret_tensor(buf706, (1, 512, 1024), (524288, 1024, 1), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf729, buf721, primals_110, mul_45, div_106, getitem_67, buf730, 512, 1024, grid=grid(512), stream=stream0)
        del div_106
        del getitem_67
        del primals_110
        buf727 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf728 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf721, mul_45, buf727, buf728, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_45
        buf731 = buf721; del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf730, (512, 1024), (1024, 1), 0), permute_838, out=buf731)
        del permute_838
        buf732 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf730, (1024, 512), (1, 1024), 0), view_148, out=buf732)
        del view_148
        buf733 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf730, buf733, 4096, 128, grid=grid(4096), stream=stream0)
        del buf730
        buf734 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf733, buf734, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf735 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf731, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_51, clone_default_52, clone_default_53, None, alias_default_35, getitem_366, getitem_367, getitem_368, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_35
        del clone_default_51
        del clone_default_52
        del clone_default_53
        del getitem_366
        del getitem_367
        del getitem_368
        buf736 = buf735[0]
        buf737 = buf735[1]
        buf738 = buf735[2]
        del buf735
        buf739 = buf731; del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf738, (512, 1024), (1024, 1), 0), permute_850, out=buf739)
        del permute_850
        buf740 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf738, (1024, 512), (1, 1024), 0), view_132, out=buf740)
        buf741 = buf733; del buf733  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf738, buf741, 4096, 128, grid=grid(4096), stream=stream0)
        buf742 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf741, buf742, 1024, 4, grid=grid(1024), stream=stream0)
        buf743 = reinterpret_tensor(buf738, (512, 1024), (1024, 1), 0); del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf737, (512, 1024), (1024, 1), 0), permute_855, out=buf743)
        del permute_855
        buf744 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf737, (1024, 512), (1, 1024), 0), view_132, out=buf744)
        buf745 = buf741; del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf737, buf745, 4096, 128, grid=grid(4096), stream=stream0)
        buf746 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf745, buf746, 1024, 4, grid=grid(1024), stream=stream0)
        buf747 = reinterpret_tensor(buf737, (512, 1024), (1024, 1), 0); del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf736, (512, 1024), (1024, 1), 0), permute_859, out=buf747)
        del permute_859
        buf748 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf736, (1024, 512), (1, 1024), 0), view_132, out=buf748)
        del view_132
        buf749 = buf745; del buf745  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf736, buf749, 4096, 128, grid=grid(4096), stream=stream0)
        buf750 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf749, buf750, 1024, 4, grid=grid(1024), stream=stream0)
        buf755 = buf729; del buf729  # reuse
        buf756 = reinterpret_tensor(buf736, (1, 512, 1024), (524288, 1024, 1), 0); del buf736  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf755, buf739, buf743, buf747, primals_100, mul_43, div_108, getitem_61, buf756, 512, 1024, grid=grid(512), stream=stream0)
        del div_108
        del getitem_61
        del primals_100
        buf753 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf754 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf739, buf743, buf747, mul_43, buf753, buf754, 1024, 512, grid=grid(1024), stream=stream0)
        del buf739
        del buf743
        del mul_43
        buf757 = reinterpret_tensor(buf720, (512, 4096), (4096, 1), 0); del buf720  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf756, (512, 1024), (1024, 1), 0), permute_863, out=buf757)
        del permute_863
        buf758 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf756, (1024, 512), (1, 1024), 0), view_130, out=buf758)
        del view_130
        buf759 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf756, buf759, 4096, 128, grid=grid(4096), stream=stream0)
        buf760 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf759, buf760, 1024, 4, grid=grid(1024), stream=stream0)
        buf761 = reinterpret_tensor(buf757, (1, 512, 4096), (2097152, 4096, 1), 0); del buf757  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf761, addmm_34, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_34
        buf762 = reinterpret_tensor(buf756, (512, 1024), (1024, 1), 0); del buf756  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf761, (512, 4096), (4096, 1), 0), permute_867, out=buf762)
        del permute_867
        buf763 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf761, (4096, 512), (1, 4096), 0), view_128, out=buf763)
        del view_128
        buf764 = buf723; del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf761, buf764, 16384, 128, grid=grid(16384), stream=stream0)
        buf765 = reinterpret_tensor(buf759, (1, 4096), (4096, 1), 0); del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf764, buf765, 4096, 4, grid=grid(4096), stream=stream0)
        buf770 = buf755; del buf755  # reuse
        buf771 = reinterpret_tensor(buf747, (1, 512, 1024), (524288, 1024, 1), 0); del buf747  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf770, buf762, primals_94, mul_38, div_109, getitem_57, buf771, 512, 1024, grid=grid(512), stream=stream0)
        del div_109
        del getitem_57
        del primals_94
        buf768 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf769 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf762, mul_38, buf768, buf769, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_38
        buf772 = buf762; del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf771, (512, 1024), (1024, 1), 0), permute_871, out=buf772)
        del permute_871
        buf773 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf771, (1024, 512), (1, 1024), 0), view_126, out=buf773)
        del view_126
        buf774 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf771, buf774, 4096, 128, grid=grid(4096), stream=stream0)
        del buf771
        buf775 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf774, buf775, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf776 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf772, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_54, clone_default_55, clone_default_56, None, alias_default_37, getitem_373, getitem_374, getitem_375, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_37
        del clone_default_54
        del clone_default_55
        del clone_default_56
        del getitem_373
        del getitem_374
        del getitem_375
        buf777 = buf776[0]
        buf778 = buf776[1]
        buf779 = buf776[2]
        del buf776
        buf780 = buf772; del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf779, (512, 1024), (1024, 1), 0), permute_883, out=buf780)
        del permute_883
        buf781 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf779, (1024, 512), (1, 1024), 0), view_110, out=buf781)
        buf782 = buf774; del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf779, buf782, 4096, 128, grid=grid(4096), stream=stream0)
        buf783 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf782, buf783, 1024, 4, grid=grid(1024), stream=stream0)
        buf784 = reinterpret_tensor(buf779, (512, 1024), (1024, 1), 0); del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf778, (512, 1024), (1024, 1), 0), permute_888, out=buf784)
        del permute_888
        buf785 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf778, (1024, 512), (1, 1024), 0), view_110, out=buf785)
        buf786 = buf782; del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf778, buf786, 4096, 128, grid=grid(4096), stream=stream0)
        buf787 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf786, buf787, 1024, 4, grid=grid(1024), stream=stream0)
        buf788 = reinterpret_tensor(buf778, (512, 1024), (1024, 1), 0); del buf778  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf777, (512, 1024), (1024, 1), 0), permute_892, out=buf788)
        del permute_892
        buf789 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf777, (1024, 512), (1, 1024), 0), view_110, out=buf789)
        del view_110
        buf790 = buf786; del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf777, buf790, 4096, 128, grid=grid(4096), stream=stream0)
        buf791 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf790, buf791, 1024, 4, grid=grid(1024), stream=stream0)
        buf796 = buf770; del buf770  # reuse
        buf797 = reinterpret_tensor(buf777, (1, 512, 1024), (524288, 1024, 1), 0); del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf796, buf780, buf784, buf788, primals_84, mul_36, div_111, getitem_51, buf797, 512, 1024, grid=grid(512), stream=stream0)
        del div_111
        del getitem_51
        del primals_84
        buf794 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf795 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf780, buf784, buf788, mul_36, buf794, buf795, 1024, 512, grid=grid(1024), stream=stream0)
        del buf780
        del buf784
        del mul_36
        buf798 = reinterpret_tensor(buf761, (512, 4096), (4096, 1), 0); del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf797, (512, 1024), (1024, 1), 0), permute_896, out=buf798)
        del permute_896
        buf799 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf797, (1024, 512), (1, 1024), 0), view_108, out=buf799)
        del view_108
        buf800 = buf790; del buf790  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf797, buf800, 4096, 128, grid=grid(4096), stream=stream0)
        buf801 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf800, buf801, 1024, 4, grid=grid(1024), stream=stream0)
        buf802 = reinterpret_tensor(buf798, (1, 512, 4096), (2097152, 4096, 1), 0); del buf798  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf802, addmm_28, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_28
        buf803 = reinterpret_tensor(buf797, (512, 1024), (1024, 1), 0); del buf797  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf802, (512, 4096), (4096, 1), 0), permute_900, out=buf803)
        del permute_900
        buf804 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf802, (4096, 512), (1, 4096), 0), view_106, out=buf804)
        del view_106
        buf805 = buf764; del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf802, buf805, 16384, 128, grid=grid(16384), stream=stream0)
        buf806 = reinterpret_tensor(buf800, (1, 4096), (4096, 1), 0); del buf800  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf805, buf806, 4096, 4, grid=grid(4096), stream=stream0)
        buf811 = buf796; del buf796  # reuse
        buf812 = reinterpret_tensor(buf788, (1, 512, 1024), (524288, 1024, 1), 0); del buf788  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf811, buf803, primals_78, mul_31, div_112, getitem_47, buf812, 512, 1024, grid=grid(512), stream=stream0)
        del div_112
        del getitem_47
        del primals_78
        buf809 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf810 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf803, mul_31, buf809, buf810, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_31
        buf813 = buf803; del buf803  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf812, (512, 1024), (1024, 1), 0), permute_904, out=buf813)
        del permute_904
        buf814 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf812, (1024, 512), (1, 1024), 0), view_104, out=buf814)
        del view_104
        buf815 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf812, buf815, 4096, 128, grid=grid(4096), stream=stream0)
        del buf812
        buf816 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf815, buf816, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf817 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf813, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_57, clone_default_58, clone_default_59, None, alias_default_39, getitem_380, getitem_381, getitem_382, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_39
        del clone_default_57
        del clone_default_58
        del clone_default_59
        del getitem_380
        del getitem_381
        del getitem_382
        buf818 = buf817[0]
        buf819 = buf817[1]
        buf820 = buf817[2]
        del buf817
        buf821 = buf813; del buf813  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf820, (512, 1024), (1024, 1), 0), permute_916, out=buf821)
        del permute_916
        buf822 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf820, (1024, 512), (1, 1024), 0), view_88, out=buf822)
        buf823 = buf815; del buf815  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf820, buf823, 4096, 128, grid=grid(4096), stream=stream0)
        buf824 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf823, buf824, 1024, 4, grid=grid(1024), stream=stream0)
        buf825 = reinterpret_tensor(buf820, (512, 1024), (1024, 1), 0); del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf819, (512, 1024), (1024, 1), 0), permute_921, out=buf825)
        del permute_921
        buf826 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf819, (1024, 512), (1, 1024), 0), view_88, out=buf826)
        buf827 = buf823; del buf823  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf819, buf827, 4096, 128, grid=grid(4096), stream=stream0)
        buf828 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf827, buf828, 1024, 4, grid=grid(1024), stream=stream0)
        buf829 = reinterpret_tensor(buf819, (512, 1024), (1024, 1), 0); del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf818, (512, 1024), (1024, 1), 0), permute_925, out=buf829)
        del permute_925
        buf830 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf818, (1024, 512), (1, 1024), 0), view_88, out=buf830)
        del view_88
        buf831 = buf827; del buf827  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf818, buf831, 4096, 128, grid=grid(4096), stream=stream0)
        buf832 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf831, buf832, 1024, 4, grid=grid(1024), stream=stream0)
        buf837 = buf811; del buf811  # reuse
        buf838 = reinterpret_tensor(buf818, (1, 512, 1024), (524288, 1024, 1), 0); del buf818  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf837, buf821, buf825, buf829, primals_68, mul_29, div_114, getitem_41, buf838, 512, 1024, grid=grid(512), stream=stream0)
        del div_114
        del getitem_41
        del primals_68
        buf835 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf836 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf821, buf825, buf829, mul_29, buf835, buf836, 1024, 512, grid=grid(1024), stream=stream0)
        del buf821
        del buf825
        del mul_29
        buf839 = reinterpret_tensor(buf802, (512, 4096), (4096, 1), 0); del buf802  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf838, (512, 1024), (1024, 1), 0), permute_929, out=buf839)
        del permute_929
        buf840 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf838, (1024, 512), (1, 1024), 0), view_86, out=buf840)
        del view_86
        buf841 = buf831; del buf831  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf838, buf841, 4096, 128, grid=grid(4096), stream=stream0)
        buf842 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf841, buf842, 1024, 4, grid=grid(1024), stream=stream0)
        buf843 = reinterpret_tensor(buf839, (1, 512, 4096), (2097152, 4096, 1), 0); del buf839  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf843, addmm_22, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_22
        buf844 = reinterpret_tensor(buf838, (512, 1024), (1024, 1), 0); del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf843, (512, 4096), (4096, 1), 0), permute_933, out=buf844)
        del permute_933
        buf845 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf843, (4096, 512), (1, 4096), 0), view_84, out=buf845)
        del view_84
        buf846 = buf805; del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf843, buf846, 16384, 128, grid=grid(16384), stream=stream0)
        buf847 = reinterpret_tensor(buf841, (1, 4096), (4096, 1), 0); del buf841  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf846, buf847, 4096, 4, grid=grid(4096), stream=stream0)
        buf852 = buf837; del buf837  # reuse
        buf853 = reinterpret_tensor(buf829, (1, 512, 1024), (524288, 1024, 1), 0); del buf829  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf852, buf844, primals_62, mul_24, div_115, getitem_37, buf853, 512, 1024, grid=grid(512), stream=stream0)
        del div_115
        del getitem_37
        del primals_62
        buf850 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf851 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf844, mul_24, buf850, buf851, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_24
        buf854 = buf844; del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf853, (512, 1024), (1024, 1), 0), permute_937, out=buf854)
        del permute_937
        buf855 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf853, (1024, 512), (1, 1024), 0), view_82, out=buf855)
        del view_82
        buf856 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf853, buf856, 4096, 128, grid=grid(4096), stream=stream0)
        del buf853
        buf857 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf856, buf857, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf858 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf854, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_60, clone_default_61, clone_default_62, None, alias_default_41, getitem_387, getitem_388, getitem_389, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_41
        del clone_default_60
        del clone_default_61
        del clone_default_62
        del getitem_387
        del getitem_388
        del getitem_389
        buf859 = buf858[0]
        buf860 = buf858[1]
        buf861 = buf858[2]
        del buf858
        buf862 = buf854; del buf854  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf861, (512, 1024), (1024, 1), 0), permute_949, out=buf862)
        del permute_949
        buf863 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf861, (1024, 512), (1, 1024), 0), view_66, out=buf863)
        buf864 = buf856; del buf856  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf861, buf864, 4096, 128, grid=grid(4096), stream=stream0)
        buf865 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf864, buf865, 1024, 4, grid=grid(1024), stream=stream0)
        buf866 = reinterpret_tensor(buf861, (512, 1024), (1024, 1), 0); del buf861  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf860, (512, 1024), (1024, 1), 0), permute_954, out=buf866)
        del permute_954
        buf867 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf860, (1024, 512), (1, 1024), 0), view_66, out=buf867)
        buf868 = buf864; del buf864  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf860, buf868, 4096, 128, grid=grid(4096), stream=stream0)
        buf869 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf868, buf869, 1024, 4, grid=grid(1024), stream=stream0)
        buf870 = reinterpret_tensor(buf860, (512, 1024), (1024, 1), 0); del buf860  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf859, (512, 1024), (1024, 1), 0), permute_958, out=buf870)
        del permute_958
        buf871 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf859, (1024, 512), (1, 1024), 0), view_66, out=buf871)
        del view_66
        buf872 = buf868; del buf868  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf859, buf872, 4096, 128, grid=grid(4096), stream=stream0)
        buf873 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf872, buf873, 1024, 4, grid=grid(1024), stream=stream0)
        buf878 = buf852; del buf852  # reuse
        buf879 = reinterpret_tensor(buf859, (1, 512, 1024), (524288, 1024, 1), 0); del buf859  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf878, buf862, buf866, buf870, primals_52, mul_22, div_117, getitem_31, buf879, 512, 1024, grid=grid(512), stream=stream0)
        del div_117
        del getitem_31
        del primals_52
        buf876 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf877 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf862, buf866, buf870, mul_22, buf876, buf877, 1024, 512, grid=grid(1024), stream=stream0)
        del buf862
        del buf866
        del mul_22
        buf880 = reinterpret_tensor(buf843, (512, 4096), (4096, 1), 0); del buf843  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf879, (512, 1024), (1024, 1), 0), permute_962, out=buf880)
        del permute_962
        buf881 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf879, (1024, 512), (1, 1024), 0), view_64, out=buf881)
        del view_64
        buf882 = buf872; del buf872  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf879, buf882, 4096, 128, grid=grid(4096), stream=stream0)
        buf883 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf882, buf883, 1024, 4, grid=grid(1024), stream=stream0)
        buf884 = reinterpret_tensor(buf880, (1, 512, 4096), (2097152, 4096, 1), 0); del buf880  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf884, addmm_16, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_16
        buf885 = reinterpret_tensor(buf879, (512, 1024), (1024, 1), 0); del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf884, (512, 4096), (4096, 1), 0), permute_966, out=buf885)
        del permute_966
        buf886 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf884, (4096, 512), (1, 4096), 0), view_62, out=buf886)
        del view_62
        buf887 = buf846; del buf846  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf884, buf887, 16384, 128, grid=grid(16384), stream=stream0)
        buf888 = reinterpret_tensor(buf882, (1, 4096), (4096, 1), 0); del buf882  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf887, buf888, 4096, 4, grid=grid(4096), stream=stream0)
        buf893 = buf878; del buf878  # reuse
        buf894 = reinterpret_tensor(buf870, (1, 512, 1024), (524288, 1024, 1), 0); del buf870  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf893, buf885, primals_46, mul_17, div_118, getitem_27, buf894, 512, 1024, grid=grid(512), stream=stream0)
        del div_118
        del getitem_27
        del primals_46
        buf891 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf892 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf885, mul_17, buf891, buf892, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_17
        buf895 = buf885; del buf885  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf894, (512, 1024), (1024, 1), 0), permute_970, out=buf895)
        del permute_970
        buf896 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf894, (1024, 512), (1, 1024), 0), view_60, out=buf896)
        del view_60
        buf897 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf894, buf897, 4096, 128, grid=grid(4096), stream=stream0)
        del buf894
        buf898 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf897, buf898, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf899 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf895, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_63, clone_default_64, clone_default_65, None, alias_default_43, getitem_394, getitem_395, getitem_396, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_43
        del clone_default_63
        del clone_default_64
        del clone_default_65
        del getitem_394
        del getitem_395
        del getitem_396
        buf900 = buf899[0]
        buf901 = buf899[1]
        buf902 = buf899[2]
        del buf899
        buf903 = buf895; del buf895  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf902, (512, 1024), (1024, 1), 0), permute_982, out=buf903)
        del permute_982
        buf904 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf902, (1024, 512), (1, 1024), 0), view_44, out=buf904)
        buf905 = buf897; del buf897  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf902, buf905, 4096, 128, grid=grid(4096), stream=stream0)
        buf906 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf905, buf906, 1024, 4, grid=grid(1024), stream=stream0)
        buf907 = reinterpret_tensor(buf902, (512, 1024), (1024, 1), 0); del buf902  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf901, (512, 1024), (1024, 1), 0), permute_987, out=buf907)
        del permute_987
        buf908 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf901, (1024, 512), (1, 1024), 0), view_44, out=buf908)
        buf909 = buf905; del buf905  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf901, buf909, 4096, 128, grid=grid(4096), stream=stream0)
        buf910 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf909, buf910, 1024, 4, grid=grid(1024), stream=stream0)
        buf911 = reinterpret_tensor(buf901, (512, 1024), (1024, 1), 0); del buf901  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf900, (512, 1024), (1024, 1), 0), permute_991, out=buf911)
        del permute_991
        buf912 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf900, (1024, 512), (1, 1024), 0), view_44, out=buf912)
        del view_44
        buf913 = buf909; del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf900, buf913, 4096, 128, grid=grid(4096), stream=stream0)
        buf914 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf913, buf914, 1024, 4, grid=grid(1024), stream=stream0)
        buf919 = buf893; del buf893  # reuse
        buf920 = reinterpret_tensor(buf900, (1, 512, 1024), (524288, 1024, 1), 0); del buf900  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf919, buf903, buf907, buf911, primals_36, mul_15, div_120, getitem_21, buf920, 512, 1024, grid=grid(512), stream=stream0)
        del div_120
        del getitem_21
        del primals_36
        buf917 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf918 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf903, buf907, buf911, mul_15, buf917, buf918, 1024, 512, grid=grid(1024), stream=stream0)
        del buf903
        del buf907
        del mul_15
        buf921 = reinterpret_tensor(buf884, (512, 4096), (4096, 1), 0); del buf884  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf920, (512, 1024), (1024, 1), 0), permute_995, out=buf921)
        del permute_995
        buf922 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf920, (1024, 512), (1, 1024), 0), view_42, out=buf922)
        del view_42
        buf923 = buf913; del buf913  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf920, buf923, 4096, 128, grid=grid(4096), stream=stream0)
        buf924 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf923, buf924, 1024, 4, grid=grid(1024), stream=stream0)
        buf925 = reinterpret_tensor(buf921, (1, 512, 4096), (2097152, 4096, 1), 0); del buf921  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf925, addmm_10, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_10
        buf926 = reinterpret_tensor(buf920, (512, 1024), (1024, 1), 0); del buf920  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf925, (512, 4096), (4096, 1), 0), permute_999, out=buf926)
        del permute_999
        buf927 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf925, (4096, 512), (1, 4096), 0), view_40, out=buf927)
        del view_40
        buf928 = buf887; del buf887  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf925, buf928, 16384, 128, grid=grid(16384), stream=stream0)
        buf929 = reinterpret_tensor(buf923, (1, 4096), (4096, 1), 0); del buf923  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf928, buf929, 4096, 4, grid=grid(4096), stream=stream0)
        buf934 = buf919; del buf919  # reuse
        buf935 = reinterpret_tensor(buf911, (1, 512, 1024), (524288, 1024, 1), 0); del buf911  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf934, buf926, primals_30, mul_10, div_121, getitem_17, buf935, 512, 1024, grid=grid(512), stream=stream0)
        del div_121
        del getitem_17
        del primals_30
        buf932 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf933 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf926, mul_10, buf932, buf933, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_10
        buf936 = buf926; del buf926  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf935, (512, 1024), (1024, 1), 0), permute_1003, out=buf936)
        del permute_1003
        buf937 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf935, (1024, 512), (1, 1024), 0), view_38, out=buf937)
        del view_38
        buf938 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf935, buf938, 4096, 128, grid=grid(4096), stream=stream0)
        del buf935
        buf939 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf938, buf939, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf940 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf936, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_66, clone_default_67, clone_default_68, None, alias_default_45, getitem_401, getitem_402, getitem_403, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_45
        del clone_default_66
        del clone_default_67
        del clone_default_68
        del getitem_401
        del getitem_402
        del getitem_403
        buf941 = buf940[0]
        buf942 = buf940[1]
        buf943 = buf940[2]
        del buf940
        buf944 = buf936; del buf936  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf943, (512, 1024), (1024, 1), 0), permute_1015, out=buf944)
        del permute_1015
        buf945 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf943, (1024, 512), (1, 1024), 0), view_22, out=buf945)
        buf946 = buf938; del buf938  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf943, buf946, 4096, 128, grid=grid(4096), stream=stream0)
        buf947 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf946, buf947, 1024, 4, grid=grid(1024), stream=stream0)
        buf948 = reinterpret_tensor(buf943, (512, 1024), (1024, 1), 0); del buf943  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf942, (512, 1024), (1024, 1), 0), permute_1020, out=buf948)
        del permute_1020
        buf949 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf942, (1024, 512), (1, 1024), 0), view_22, out=buf949)
        buf950 = buf946; del buf946  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf942, buf950, 4096, 128, grid=grid(4096), stream=stream0)
        buf951 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf950, buf951, 1024, 4, grid=grid(1024), stream=stream0)
        buf952 = reinterpret_tensor(buf942, (512, 1024), (1024, 1), 0); del buf942  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf941, (512, 1024), (1024, 1), 0), permute_1024, out=buf952)
        del permute_1024
        buf953 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf941, (1024, 512), (1, 1024), 0), view_22, out=buf953)
        del view_22
        buf954 = buf950; del buf950  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf941, buf954, 4096, 128, grid=grid(4096), stream=stream0)
        buf955 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf954, buf955, 1024, 4, grid=grid(1024), stream=stream0)
        buf960 = buf934; del buf934  # reuse
        buf961 = reinterpret_tensor(buf941, (1, 512, 1024), (524288, 1024, 1), 0); del buf941  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_13.run(buf960, buf944, buf948, buf952, primals_20, mul_8, div_123, getitem_11, buf961, 512, 1024, grid=grid(512), stream=stream0)
        del div_123
        del getitem_11
        del primals_20
        buf958 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf959 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf944, buf948, buf952, mul_8, buf958, buf959, 1024, 512, grid=grid(1024), stream=stream0)
        del buf944
        del mul_8
        buf962 = reinterpret_tensor(buf925, (512, 4096), (4096, 1), 0); del buf925  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf961, (512, 1024), (1024, 1), 0), permute_1028, out=buf962)
        del permute_1028
        buf963 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf961, (1024, 512), (1, 1024), 0), view_20, out=buf963)
        del view_20
        buf964 = buf954; del buf954  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf961, buf964, 4096, 128, grid=grid(4096), stream=stream0)
        buf965 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf964, buf965, 1024, 4, grid=grid(1024), stream=stream0)
        buf966 = reinterpret_tensor(buf962, (1, 512, 4096), (2097152, 4096, 1), 0); del buf962  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf966, addmm_4, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_4
        buf967 = reinterpret_tensor(buf961, (512, 1024), (1024, 1), 0); del buf961  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf966, (512, 4096), (4096, 1), 0), permute_1032, out=buf967)
        del permute_1032
        buf968 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf966, (4096, 512), (1, 4096), 0), view_18, out=buf968)
        del view_18
        buf969 = buf928; del buf928  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf966, buf969, 16384, 128, grid=grid(16384), stream=stream0)
        del buf966
        buf970 = reinterpret_tensor(buf964, (1, 4096), (4096, 1), 0); del buf964  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf969, buf970, 4096, 4, grid=grid(4096), stream=stream0)
        del buf969
        buf975 = buf960; del buf960  # reuse
        buf976 = reinterpret_tensor(buf952, (1, 512, 1024), (524288, 1024, 1), 0); del buf952  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf975, buf967, primals_14, mul_3, div_124, getitem_7, buf976, 512, 1024, grid=grid(512), stream=stream0)
        del div_124
        del getitem_7
        del primals_14
        buf973 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf974 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf967, mul_3, buf973, buf974, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_3
        buf977 = buf967; del buf967  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf976, (512, 1024), (1024, 1), 0), permute_1036, out=buf977)
        del permute_1036
        buf978 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf976, (1024, 512), (1, 1024), 0), view_16, out=buf978)
        del view_16
        buf979 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf976, buf979, 4096, 128, grid=grid(4096), stream=stream0)
        buf980 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf979, buf980, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf981 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf977, (1, 16, 512, 64), (524288, 64, 1024, 1), 0), clone_default_69, clone_default_70, clone_default_71, None, alias_default_47, getitem_408, getitem_409, getitem_410, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_47
        del clone_default_69
        del clone_default_70
        del clone_default_71
        del getitem_408
        del getitem_409
        del getitem_410
        buf982 = buf981[0]
        buf983 = buf981[1]
        buf984 = buf981[2]
        del buf981
        buf985 = buf977; del buf977  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf984, (512, 1024), (1024, 1), 0), permute_1048, out=buf985)
        del permute_1048
        buf986 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf984, (1024, 512), (1, 1024), 0), view, out=buf986)
        buf987 = buf979; del buf979  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf984, buf987, 4096, 128, grid=grid(4096), stream=stream0)
        buf988 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf987, buf988, 1024, 4, grid=grid(1024), stream=stream0)
        buf989 = reinterpret_tensor(buf984, (512, 1024), (1024, 1), 0); del buf984  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf983, (512, 1024), (1024, 1), 0), permute_1053, out=buf989)
        del permute_1053
        buf990 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf983, (1024, 512), (1, 1024), 0), view, out=buf990)
        buf991 = buf987; del buf987  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf983, buf991, 4096, 128, grid=grid(4096), stream=stream0)
        buf992 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf991, buf992, 1024, 4, grid=grid(1024), stream=stream0)
        buf993 = reinterpret_tensor(buf983, (512, 1024), (1024, 1), 0); del buf983  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf982, (512, 1024), (1024, 1), 0), permute_1057, out=buf993)
        del permute_1057
        buf994 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf982, (1024, 512), (1, 1024), 0), view, out=buf994)
        del view
        buf995 = buf991; del buf991  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf982, buf995, 4096, 128, grid=grid(4096), stream=stream0)
        buf996 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf995, buf996, 1024, 4, grid=grid(1024), stream=stream0)
        del buf995
        buf1001 = buf975; del buf975  # reuse
        buf1003 = reinterpret_tensor(buf982, (1, 512, 1024), (524288, 1024, 1), 0); del buf982  # reuse
        buf1007 = buf976; del buf976  # reuse
        buf1011 = reinterpret_tensor(buf948, (1, 512, 1024), (524288, 1024, 1), 0); del buf948  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_15.run(buf1001, buf985, buf989, buf993, primals_4, mul_1, div_126, slice_3, getitem_1, primals_393, buf1003, buf1007, buf1011, 512, 1024, grid=grid(512), stream=stream0)
        del buf1001
        del div_126
        del getitem_1
        del primals_4
        buf999 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1000 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf985, buf989, buf993, mul_1, buf999, buf1000, 1024, 512, grid=grid(1024), stream=stream0)
        del buf985
        del buf989
        del mul_1
        buf1002 = buf993; del buf993  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf1002, 524288, grid=grid(524288), stream=stream0)
        aten.index_put_(buf1002, [slice_3], buf1003, True)
        del buf1003
        del slice_3
        buf1006 = empty((2, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_17.run(buf1006, 2048, grid=grid(2048), stream=stream0)
        aten.index_put_(buf1006, [full_default], buf1007, True)
        del buf1007
        del full_default
        buf1010 = empty((29056, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_18.run(buf1010, 29753344, grid=grid(29753344), stream=stream0)
        aten.index_put_(buf1010, [primals_393], buf1011, True)
        del buf1011
        del primals_393
        return (buf1010, buf1006, buf1002, buf999, buf1000, reinterpret_tensor(buf994, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf996, (1024, ), (1, ), 0), reinterpret_tensor(buf990, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf992, (1024, ), (1, ), 0), reinterpret_tensor(buf986, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf988, (1024, ), (1, ), 0), reinterpret_tensor(buf978, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf980, (1024, ), (1, ), 0), buf973, buf974, reinterpret_tensor(buf968, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf970, (4096, ), (1, ), 0), reinterpret_tensor(buf963, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf965, (1024, ), (1, ), 0), buf958, buf959, reinterpret_tensor(buf953, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf955, (1024, ), (1, ), 0), reinterpret_tensor(buf949, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf951, (1024, ), (1, ), 0), reinterpret_tensor(buf945, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf947, (1024, ), (1, ), 0), reinterpret_tensor(buf937, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf939, (1024, ), (1, ), 0), buf932, buf933, reinterpret_tensor(buf927, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf929, (4096, ), (1, ), 0), reinterpret_tensor(buf922, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf924, (1024, ), (1, ), 0), buf917, buf918, reinterpret_tensor(buf912, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf914, (1024, ), (1, ), 0), reinterpret_tensor(buf908, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf910, (1024, ), (1, ), 0), reinterpret_tensor(buf904, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf906, (1024, ), (1, ), 0), reinterpret_tensor(buf896, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf898, (1024, ), (1, ), 0), buf891, buf892, reinterpret_tensor(buf886, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf888, (4096, ), (1, ), 0), reinterpret_tensor(buf881, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf883, (1024, ), (1, ), 0), buf876, buf877, reinterpret_tensor(buf871, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf873, (1024, ), (1, ), 0), reinterpret_tensor(buf867, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf869, (1024, ), (1, ), 0), reinterpret_tensor(buf863, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf865, (1024, ), (1, ), 0), reinterpret_tensor(buf855, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf857, (1024, ), (1, ), 0), buf850, buf851, reinterpret_tensor(buf845, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf847, (4096, ), (1, ), 0), reinterpret_tensor(buf840, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf842, (1024, ), (1, ), 0), buf835, buf836, reinterpret_tensor(buf830, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf832, (1024, ), (1, ), 0), reinterpret_tensor(buf826, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf828, (1024, ), (1, ), 0), reinterpret_tensor(buf822, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf824, (1024, ), (1, ), 0), reinterpret_tensor(buf814, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf816, (1024, ), (1, ), 0), buf809, buf810, reinterpret_tensor(buf804, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf806, (4096, ), (1, ), 0), reinterpret_tensor(buf799, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf801, (1024, ), (1, ), 0), buf794, buf795, reinterpret_tensor(buf789, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf791, (1024, ), (1, ), 0), reinterpret_tensor(buf785, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf787, (1024, ), (1, ), 0), reinterpret_tensor(buf781, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf783, (1024, ), (1, ), 0), reinterpret_tensor(buf773, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf775, (1024, ), (1, ), 0), buf768, buf769, reinterpret_tensor(buf763, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf765, (4096, ), (1, ), 0), reinterpret_tensor(buf758, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf760, (1024, ), (1, ), 0), buf753, buf754, reinterpret_tensor(buf748, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf750, (1024, ), (1, ), 0), reinterpret_tensor(buf744, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf746, (1024, ), (1, ), 0), reinterpret_tensor(buf740, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf742, (1024, ), (1, ), 0), reinterpret_tensor(buf732, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf734, (1024, ), (1, ), 0), buf727, buf728, reinterpret_tensor(buf722, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf724, (4096, ), (1, ), 0), reinterpret_tensor(buf717, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf719, (1024, ), (1, ), 0), buf712, buf713, reinterpret_tensor(buf707, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf709, (1024, ), (1, ), 0), reinterpret_tensor(buf703, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf705, (1024, ), (1, ), 0), reinterpret_tensor(buf699, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf701, (1024, ), (1, ), 0), reinterpret_tensor(buf691, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf693, (1024, ), (1, ), 0), buf686, buf687, reinterpret_tensor(buf681, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf683, (4096, ), (1, ), 0), reinterpret_tensor(buf676, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf678, (1024, ), (1, ), 0), buf671, buf672, reinterpret_tensor(buf666, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf668, (1024, ), (1, ), 0), reinterpret_tensor(buf662, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf664, (1024, ), (1, ), 0), reinterpret_tensor(buf658, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf660, (1024, ), (1, ), 0), reinterpret_tensor(buf650, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf652, (1024, ), (1, ), 0), buf645, buf646, reinterpret_tensor(buf640, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf642, (4096, ), (1, ), 0), reinterpret_tensor(buf635, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf637, (1024, ), (1, ), 0), buf630, buf631, reinterpret_tensor(buf625, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf627, (1024, ), (1, ), 0), reinterpret_tensor(buf621, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf623, (1024, ), (1, ), 0), reinterpret_tensor(buf617, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf619, (1024, ), (1, ), 0), reinterpret_tensor(buf609, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf611, (1024, ), (1, ), 0), buf604, buf605, reinterpret_tensor(buf599, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf601, (4096, ), (1, ), 0), reinterpret_tensor(buf594, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf596, (1024, ), (1, ), 0), buf589, buf590, reinterpret_tensor(buf584, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf586, (1024, ), (1, ), 0), reinterpret_tensor(buf580, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf582, (1024, ), (1, ), 0), reinterpret_tensor(buf576, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf578, (1024, ), (1, ), 0), reinterpret_tensor(buf568, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf570, (1024, ), (1, ), 0), buf563, buf564, reinterpret_tensor(buf558, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf560, (4096, ), (1, ), 0), reinterpret_tensor(buf553, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf555, (1024, ), (1, ), 0), buf548, buf549, reinterpret_tensor(buf543, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf545, (1024, ), (1, ), 0), reinterpret_tensor(buf539, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf541, (1024, ), (1, ), 0), reinterpret_tensor(buf535, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf537, (1024, ), (1, ), 0), reinterpret_tensor(buf527, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf529, (1024, ), (1, ), 0), buf522, buf523, reinterpret_tensor(buf517, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf519, (4096, ), (1, ), 0), reinterpret_tensor(buf512, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf514, (1024, ), (1, ), 0), buf507, buf508, reinterpret_tensor(buf502, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf504, (1024, ), (1, ), 0), reinterpret_tensor(buf498, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf500, (1024, ), (1, ), 0), reinterpret_tensor(buf494, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf496, (1024, ), (1, ), 0), reinterpret_tensor(buf486, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf488, (1024, ), (1, ), 0), buf481, buf482, reinterpret_tensor(buf476, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf478, (4096, ), (1, ), 0), reinterpret_tensor(buf471, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf473, (1024, ), (1, ), 0), buf466, buf467, reinterpret_tensor(buf461, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf463, (1024, ), (1, ), 0), reinterpret_tensor(buf457, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf459, (1024, ), (1, ), 0), reinterpret_tensor(buf453, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf455, (1024, ), (1, ), 0), reinterpret_tensor(buf445, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf447, (1024, ), (1, ), 0), buf440, buf441, reinterpret_tensor(buf435, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf437, (4096, ), (1, ), 0), reinterpret_tensor(buf430, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf432, (1024, ), (1, ), 0), buf425, buf426, reinterpret_tensor(buf420, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf422, (1024, ), (1, ), 0), reinterpret_tensor(buf416, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf418, (1024, ), (1, ), 0), reinterpret_tensor(buf412, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf414, (1024, ), (1, ), 0), reinterpret_tensor(buf404, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf406, (1024, ), (1, ), 0), buf399, buf400, reinterpret_tensor(buf394, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf396, (4096, ), (1, ), 0), reinterpret_tensor(buf389, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf391, (1024, ), (1, ), 0), buf384, buf385, reinterpret_tensor(buf379, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf381, (1024, ), (1, ), 0), reinterpret_tensor(buf375, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf377, (1024, ), (1, ), 0), reinterpret_tensor(buf371, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf373, (1024, ), (1, ), 0), reinterpret_tensor(buf363, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf365, (1024, ), (1, ), 0), buf358, buf359, reinterpret_tensor(buf353, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf355, (4096, ), (1, ), 0), reinterpret_tensor(buf348, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf350, (1024, ), (1, ), 0), buf343, buf344, reinterpret_tensor(buf338, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf340, (1024, ), (1, ), 0), reinterpret_tensor(buf334, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf336, (1024, ), (1, ), 0), reinterpret_tensor(buf330, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf332, (1024, ), (1, ), 0), reinterpret_tensor(buf322, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf324, (1024, ), (1, ), 0), buf317, buf318, reinterpret_tensor(buf312, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf314, (4096, ), (1, ), 0), reinterpret_tensor(buf307, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf309, (1024, ), (1, ), 0), buf302, buf303, reinterpret_tensor(buf297, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf299, (1024, ), (1, ), 0), reinterpret_tensor(buf293, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf295, (1024, ), (1, ), 0), reinterpret_tensor(buf289, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf291, (1024, ), (1, ), 0), reinterpret_tensor(buf281, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf283, (1024, ), (1, ), 0), buf276, buf277, reinterpret_tensor(buf271, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf273, (4096, ), (1, ), 0), reinterpret_tensor(buf266, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf268, (1024, ), (1, ), 0), buf261, buf262, reinterpret_tensor(buf256, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf258, (1024, ), (1, ), 0), reinterpret_tensor(buf252, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf254, (1024, ), (1, ), 0), reinterpret_tensor(buf248, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf250, (1024, ), (1, ), 0), reinterpret_tensor(buf240, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf242, (1024, ), (1, ), 0), buf235, buf236, reinterpret_tensor(buf230, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf232, (4096, ), (1, ), 0), reinterpret_tensor(buf225, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf227, (1024, ), (1, ), 0), buf220, buf221, reinterpret_tensor(buf215, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf217, (1024, ), (1, ), 0), reinterpret_tensor(buf211, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf213, (1024, ), (1, ), 0), reinterpret_tensor(buf207, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf209, (1024, ), (1, ), 0), reinterpret_tensor(buf199, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf201, (1024, ), (1, ), 0), buf194, buf195, reinterpret_tensor(buf189, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf191, (4096, ), (1, ), 0), reinterpret_tensor(buf184, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf186, (1024, ), (1, ), 0), buf179, buf180, reinterpret_tensor(buf174, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf176, (1024, ), (1, ), 0), reinterpret_tensor(buf170, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf172, (1024, ), (1, ), 0), reinterpret_tensor(buf166, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf168, (1024, ), (1, ), 0), reinterpret_tensor(buf158, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf160, (1024, ), (1, ), 0), buf153, buf154, reinterpret_tensor(buf148, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf150, (4096, ), (1, ), 0), reinterpret_tensor(buf143, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf145, (1024, ), (1, ), 0), buf138, buf139, reinterpret_tensor(buf133, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf135, (1024, ), (1, ), 0), reinterpret_tensor(buf129, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf131, (1024, ), (1, ), 0), reinterpret_tensor(buf125, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf127, (1024, ), (1, ), 0), reinterpret_tensor(buf117, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf119, (1024, ), (1, ), 0), buf112, buf113, reinterpret_tensor(buf107, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf109, (4096, ), (1, ), 0), reinterpret_tensor(buf102, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf104, (1024, ), (1, ), 0), buf97, buf98, reinterpret_tensor(buf92, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf94, (1024, ), (1, ), 0), reinterpret_tensor(buf88, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf90, (1024, ), (1, ), 0), reinterpret_tensor(buf84, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf86, (1024, ), (1, ), 0), reinterpret_tensor(buf76, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf78, (1024, ), (1, ), 0), buf71, buf72, reinterpret_tensor(buf66, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf68, (4096, ), (1, ), 0), reinterpret_tensor(buf61, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf63, (1024, ), (1, ), 0), buf56, buf57, reinterpret_tensor(buf51, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf53, (1024, ), (1, ), 0), reinterpret_tensor(buf47, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf49, (1024, ), (1, ), 0), reinterpret_tensor(buf43, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf45, (1024, ), (1, ), 0), reinterpret_tensor(buf35, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf37, (1024, ), (1, ), 0), buf30, buf31, reinterpret_tensor(buf25, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf27, (4096, ), (1, ), 0), reinterpret_tensor(buf20, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf22, (1024, ), (1, ), 0), buf16, buf17, reinterpret_tensor(buf10, (2, 1024), (1024, 1), 0), reinterpret_tensor(buf12, (2, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    full_default = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    slice_3 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    getitem_1 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_1 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_69 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_70 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_71 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_408 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_409 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_410 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_47 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_3 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_66 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_67 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_68 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_401 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_402 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_403 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_45 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_10 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_15 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_63 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_64 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_65 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_394 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_395 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_396 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_43 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_17 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_22 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_60 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_61 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_62 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_387 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_388 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_389 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_41 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_29 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_57 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_58 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_59 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_380 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_381 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_382 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_39 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_31 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_36 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_54 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_55 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_56 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_373 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_374 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_375 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_37 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_38 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_43 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_51 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_52 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_53 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_366 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_367 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_368 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_35 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_45 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_50 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_48 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_49 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_50 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_359 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_360 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_361 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_33 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_52 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_57 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_45 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_46 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_47 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_352 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_353 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_354 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_31 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_59 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_64 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_42 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_43 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_44 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_345 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_346 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_347 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_29 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_66 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_71 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_39 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_40 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_41 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_338 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_339 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_340 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_27 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_236 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_73 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_78 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_36 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_37 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_38 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_331 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_332 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_333 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_25 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_258 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_80 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_85 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_264 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_33 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_34 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_35 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_324 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_325 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_326 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_23 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_280 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_87 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_282 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_76 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_284 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_131 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_92 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_286 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_30 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_31 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_32 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_317 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_318 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_319 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_21 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_302 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_137 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_94 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_304 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_82 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_306 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_141 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_99 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_308 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_27 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_28 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_29 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_310 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_312 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_19 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_324 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_147 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_101 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_326 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_88 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_328 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_151 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_106 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_330 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_24 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_25 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_26 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_303 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_304 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_17 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_346 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_157 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_108 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_348 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_94 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_350 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_161 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_113 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_352 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_21 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_22 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_23 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_296 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_298 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_15 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_368 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_167 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_115 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_370 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_100 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_372 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_171 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_120 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_374 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_18 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_19 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_20 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_289 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_13 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_390 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_177 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_122 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_392 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_106 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_394 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_181 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_127 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_396 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_15 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_16 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_17 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_282 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_283 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_11 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_412 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_187 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_129 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_414 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_112 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_416 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_191 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_134 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_418 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_12 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_13 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_14 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_275 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_277 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_9 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_434 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_197 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_136 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_436 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_118 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_438 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_201 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_141 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_440 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_9 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_10 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_11 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_268 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_269 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_7 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_456 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_207 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_143 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_458 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_124 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_460 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_211 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_148 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_462 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_6 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_7 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_8 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_261 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_262 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_263 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_5 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_478 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_217 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_150 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_480 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_130 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_482 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_221 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_155 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_484 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default_3 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_4 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_5 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_254 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_256 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_3 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_500 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_227 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_157 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_502 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_136 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_504 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_231 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_162 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_506 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_default = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_1 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_2 = rand_strided((1, 16, 512, 64), (524288, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_247 = rand_strided((1, 16, 512), (8192, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_248 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_1 = rand_strided((1, 16, 512, 64), (524288, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_522 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_237 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_164 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_524 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_142 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_526 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_241 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_169 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_528 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    sub_75 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    sub_77 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_4 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_6 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_265 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_269 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_273 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_277 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_298 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_302 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_306 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_331 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_335 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_339 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_343 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_364 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_368 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_372 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_376 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_397 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_401 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_405 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_67 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_409 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_426 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_430 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_69 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_434 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_70 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_442 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_454 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_463 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_72 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_467 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_73 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_496 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_75 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_500 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_504 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_76 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_508 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_520 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_529 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_78 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_533 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_537 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_79 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_541 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_553 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_558 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_562 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_81 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_566 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_570 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_82 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_574 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_586 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_591 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_595 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_84 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_599 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_603 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_85 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_607 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_619 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_624 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_628 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_87 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_632 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_636 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_88 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_640 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_652 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_657 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_661 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_90 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_665 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_669 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_91 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_673 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_685 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_690 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_694 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_93 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_698 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_702 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_94 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_706 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_718 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_723 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_727 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_96 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_731 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_735 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_97 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_739 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_751 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_756 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_760 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_99 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_764 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_768 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_100 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_772 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_784 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_789 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_793 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_102 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_797 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_801 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_103 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_805 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_817 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_822 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_826 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_105 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_830 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_834 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_106 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_838 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_850 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_855 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_859 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_108 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_863 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_867 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_109 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_871 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_883 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_888 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_892 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_111 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_896 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_900 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_112 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_904 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_916 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_921 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_925 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_114 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_929 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_933 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_115 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_937 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_949 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_954 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_958 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_117 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_962 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_966 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_118 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_970 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_982 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_987 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_991 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_120 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_995 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_999 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_121 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1003 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1015 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1020 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1024 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_123 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1028 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1032 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_124 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1036 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1048 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1053 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1057 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_126 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_206, primals_212, primals_222, primals_228, primals_238, primals_244, primals_254, primals_260, primals_270, primals_276, primals_286, primals_292, primals_302, primals_308, primals_318, primals_324, primals_334, primals_340, primals_350, primals_356, primals_366, primals_372, primals_382, primals_388, primals_393, full_default, slice_3, getitem_1, mul_1, view, clone_default_69, clone_default_70, clone_default_71, getitem_408, getitem_409, getitem_410, alias_default_47, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, clone_default_66, clone_default_67, clone_default_68, getitem_401, getitem_402, getitem_403, alias_default_45, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, clone_default_63, clone_default_64, clone_default_65, getitem_394, getitem_395, getitem_396, alias_default_43, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, clone_default_60, clone_default_61, clone_default_62, getitem_387, getitem_388, getitem_389, alias_default_41, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, clone_default_57, clone_default_58, clone_default_59, getitem_380, getitem_381, getitem_382, alias_default_39, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, clone_default_54, clone_default_55, clone_default_56, getitem_373, getitem_374, getitem_375, alias_default_37, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, clone_default_51, clone_default_52, clone_default_53, getitem_366, getitem_367, getitem_368, alias_default_35, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, clone_default_48, clone_default_49, clone_default_50, getitem_359, getitem_360, getitem_361, alias_default_33, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, clone_default_45, clone_default_46, clone_default_47, getitem_352, getitem_353, getitem_354, alias_default_31, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, clone_default_42, clone_default_43, clone_default_44, getitem_345, getitem_346, getitem_347, alias_default_29, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, clone_default_39, clone_default_40, clone_default_41, getitem_338, getitem_339, getitem_340, alias_default_27, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, clone_default_36, clone_default_37, clone_default_38, getitem_331, getitem_332, getitem_333, alias_default_25, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, view_264, clone_default_33, clone_default_34, clone_default_35, getitem_324, getitem_325, getitem_326, alias_default_23, view_280, getitem_127, mul_87, view_282, addmm_76, view_284, getitem_131, mul_92, view_286, clone_default_30, clone_default_31, clone_default_32, getitem_317, getitem_318, getitem_319, alias_default_21, view_302, getitem_137, mul_94, view_304, addmm_82, view_306, getitem_141, mul_99, view_308, clone_default_27, clone_default_28, clone_default_29, getitem_310, getitem_311, getitem_312, alias_default_19, view_324, getitem_147, mul_101, view_326, addmm_88, view_328, getitem_151, mul_106, view_330, clone_default_24, clone_default_25, clone_default_26, getitem_303, getitem_304, getitem_305, alias_default_17, view_346, getitem_157, mul_108, view_348, addmm_94, view_350, getitem_161, mul_113, view_352, clone_default_21, clone_default_22, clone_default_23, getitem_296, getitem_297, getitem_298, alias_default_15, view_368, getitem_167, mul_115, view_370, addmm_100, view_372, getitem_171, mul_120, view_374, clone_default_18, clone_default_19, clone_default_20, getitem_289, getitem_290, getitem_291, alias_default_13, view_390, getitem_177, mul_122, view_392, addmm_106, view_394, getitem_181, mul_127, view_396, clone_default_15, clone_default_16, clone_default_17, getitem_282, getitem_283, getitem_284, alias_default_11, view_412, getitem_187, mul_129, view_414, addmm_112, view_416, getitem_191, mul_134, view_418, clone_default_12, clone_default_13, clone_default_14, getitem_275, getitem_276, getitem_277, alias_default_9, view_434, getitem_197, mul_136, view_436, addmm_118, view_438, getitem_201, mul_141, view_440, clone_default_9, clone_default_10, clone_default_11, getitem_268, getitem_269, getitem_270, alias_default_7, view_456, getitem_207, mul_143, view_458, addmm_124, view_460, getitem_211, mul_148, view_462, clone_default_6, clone_default_7, clone_default_8, getitem_261, getitem_262, getitem_263, alias_default_5, view_478, getitem_217, mul_150, view_480, addmm_130, view_482, getitem_221, mul_155, view_484, clone_default_3, clone_default_4, clone_default_5, getitem_254, getitem_255, getitem_256, alias_default_3, view_500, getitem_227, mul_157, view_502, addmm_136, view_504, getitem_231, mul_162, view_506, clone_default, clone_default_1, clone_default_2, getitem_247, getitem_248, getitem_249, alias_default_1, view_522, getitem_237, mul_164, view_524, addmm_142, view_526, getitem_241, mul_169, view_528, sub_75, ne, sub_77, ne_3, ne_6, where_4, ne_8, where_6, permute_265, div_54, permute_269, permute_273, div_55, permute_277, permute_289, permute_294, permute_298, div_57, permute_302, permute_306, div_58, permute_310, permute_322, permute_327, permute_331, div_60, permute_335, permute_339, div_61, permute_343, permute_355, permute_360, permute_364, div_63, permute_368, permute_372, div_64, permute_376, permute_388, permute_393, permute_397, div_66, permute_401, permute_405, div_67, permute_409, permute_421, permute_426, permute_430, div_69, permute_434, permute_438, div_70, permute_442, permute_454, permute_459, permute_463, div_72, permute_467, permute_471, div_73, permute_475, permute_487, permute_492, permute_496, div_75, permute_500, permute_504, div_76, permute_508, permute_520, permute_525, permute_529, div_78, permute_533, permute_537, div_79, permute_541, permute_553, permute_558, permute_562, div_81, permute_566, permute_570, div_82, permute_574, permute_586, permute_591, permute_595, div_84, permute_599, permute_603, div_85, permute_607, permute_619, permute_624, permute_628, div_87, permute_632, permute_636, div_88, permute_640, permute_652, permute_657, permute_661, div_90, permute_665, permute_669, div_91, permute_673, permute_685, permute_690, permute_694, div_93, permute_698, permute_702, div_94, permute_706, permute_718, permute_723, permute_727, div_96, permute_731, permute_735, div_97, permute_739, permute_751, permute_756, permute_760, div_99, permute_764, permute_768, div_100, permute_772, permute_784, permute_789, permute_793, div_102, permute_797, permute_801, div_103, permute_805, permute_817, permute_822, permute_826, div_105, permute_830, permute_834, div_106, permute_838, permute_850, permute_855, permute_859, div_108, permute_863, permute_867, div_109, permute_871, permute_883, permute_888, permute_892, div_111, permute_896, permute_900, div_112, permute_904, permute_916, permute_921, permute_925, div_114, permute_929, permute_933, div_115, permute_937, permute_949, permute_954, permute_958, div_117, permute_962, permute_966, div_118, permute_970, permute_982, permute_987, permute_991, div_120, permute_995, permute_999, div_121, permute_1003, permute_1015, permute_1020, permute_1024, div_123, permute_1028, permute_1032, div_124, permute_1036, permute_1048, permute_1053, permute_1057, div_126, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MegatronBertForQuestionAnswering', benchmark_compiled_module)
