
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


# kernel path: /tmp/torchinductor_youkaichao/li/cli4idcr3cfwhbutwkeg4tc5k6lpqylrp4gzv7t3y364ffazts2m.py
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
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (r1 + (256*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 256.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp19 * tmp23
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qn/cqndyrly4dzv5kznhj62jsj25ribsya6ulcnztqmeuwokf5ypz3z.py
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
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5psng475zze6sp662mdxqoq62xz4wc4bspevot3chp4miepooy.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2ov27g67sfiqwt6sldqwyztve5dx3lvr62zpyugfg26nnmpapl4.py
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
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmi7rg3xfy36fxgfyhk5t5p66nlncfjmnppjn6objiwxvurgwt3.py
# Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_11 => add_96, erf_11, mul_83
triton_poi_fused_gelu_gelu_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: /tmp/torchinductor_youkaichao/4o/c4o5v6n4vyxdavf476y75347blyv4djtcpk2fjwacktqhr33iahr.py
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


# kernel path: /tmp/torchinductor_youkaichao/7y/c7y6bwh6kcs6uu6fdvrk74copelon3fwiizoeczrldwbowo3323l.py
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
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_11', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqhdv7n32qh663ntsvto3iujqtznokkyb65skyv3ner3vq3myb4.py
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
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r1 + (256*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 256.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w3/cw3l7f4o5cwcwkiintramm7qsjhlfsuutlwhcbsj62nifi2veo3u.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhjgrnpyibau2zgyaabuc6t3hwqjm4pwa2o3vppmcvounswknea.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (r1 + (256*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 256.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.1111111111111112
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (256*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzcebrolsm5ghext4pjou7q3cqlty3slxvb6tjfep2fzldet53g.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yi/cyidjmocmkh7b4gag7pa3qpdzwve27kl6acuhawnywx6khar4mdf.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vzxkfytz2ku5kd5hmvo5sio652y2adllayzpvy4k3cn3rwfflt.py
# Source Nodes: [start_loss], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# start_loss => full_default_2
triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*i64', 7: '*i64', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp7 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = 128.0
    tmp20 = tmp7 * tmp19
    tmp21 = tmp20 - tmp11
    tmp22 = tmp12 * tmp17
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp26 = tl.full([1, 1], -1, tl.int64)
    tmp27 = tmp25 == tmp26
    tmp28 = 0.0
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tmp31 = tmp30 == tmp26
    tmp32 = tl.where(tmp31, tmp28, tmp24)
    tmp34 = tl.full([1, 1], 0, tl.int64)
    tmp35 = tmp33 == tmp34
    tmp36 = tl.where(tmp35, tmp28, tmp24)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (128*x0)), tmp32, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (128*x0)), tmp36, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co473t3ukvzosu37gkihl2axcoh5ib77hnq2evh2em7bzrgoaoil.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6ns5amtl4uw4iwdr6scnpbeuigulf4l27vrhx7y4gupebcipn6n.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_19', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gl3rrlukpdumkeh34wnm4zotggojje4ex2lpuz7yw3wbrk3wjn.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_20', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gy/cgyy5xxv6frgzgcnmpdauh3hujinpa2xeortn5ne4oknrcg6ymse.py
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
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3906816
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
    primals_4, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, primals_112, primals_118, primals_128, primals_134, primals_144, primals_150, primals_160, primals_166, primals_176, primals_182, primals_192, primals_198, primals_204, expand, slice_4, mul_1, getitem_3, view, view_2, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_18, getitem_7, mul_3, view_20, addmm_5, view_22, getitem_11, mul_8, view_24, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_40, getitem_17, mul_10, view_42, addmm_11, view_44, getitem_21, mul_15, view_46, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_62, getitem_27, mul_17, view_64, addmm_17, view_66, getitem_31, mul_22, view_68, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_84, getitem_37, mul_24, view_86, addmm_23, view_88, getitem_41, mul_29, view_90, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_106, getitem_47, mul_31, view_108, addmm_29, view_110, getitem_51, mul_36, view_112, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_128, getitem_57, mul_38, view_130, addmm_35, view_132, getitem_61, mul_43, view_134, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_150, getitem_67, mul_45, view_152, addmm_41, view_154, getitem_71, mul_50, view_156, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_172, getitem_77, mul_52, view_174, addmm_47, view_176, getitem_81, mul_57, view_178, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_194, getitem_87, mul_59, view_196, addmm_53, view_198, getitem_91, mul_64, view_200, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_216, getitem_97, mul_66, view_218, addmm_59, view_220, getitem_101, mul_71, view_222, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_238, getitem_107, mul_73, view_240, addmm_65, view_242, getitem_111, mul_78, view_244, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_260, getitem_117, mul_80, view_262, addmm_71, view_264, getitem_121, mul_85, view_266, sub_39, ne, sub_41, ne_3, ne_6, where_4, ne_8, where_6, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_158, permute_163, permute_167, div_33, permute_171, permute_175, div_34, permute_179, permute_191, permute_196, permute_200, div_36, permute_204, permute_208, div_37, permute_212, permute_224, permute_229, permute_233, div_39, permute_237, permute_241, div_40, permute_245, permute_257, permute_262, permute_266, div_42, permute_270, permute_274, div_43, permute_278, permute_290, permute_295, permute_299, div_45, permute_303, permute_307, div_46, permute_311, permute_323, permute_328, permute_332, div_48, permute_336, permute_340, div_49, permute_344, permute_356, permute_361, permute_365, div_51, permute_369, permute_373, div_52, permute_377, permute_389, permute_394, permute_398, div_54, permute_402, permute_406, div_55, permute_410, permute_422, permute_427, permute_431, div_57, permute_435, permute_439, div_58, permute_443, permute_455, permute_460, permute_464, div_60, permute_468, permute_472, div_61, permute_476, permute_488, permute_493, permute_497, div_63, permute_501, permute_505, div_64, permute_509, permute_521, permute_526, permute_530, permute_534, div_66, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_150, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_166, (256, ), (1, ))
    assert_size_stride(primals_176, (256, ), (1, ))
    assert_size_stride(primals_182, (256, ), (1, ))
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_204, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (512, 1))
    assert_size_stride(slice_4, (1, 512), (512, 1))
    assert_size_stride(mul_1, (1, 512, 128), (65536, 128, 1))
    assert_size_stride(getitem_3, (1, 512, 128), (65536, 128, 1))
    assert_size_stride(view, (512, 128), (128, 1))
    assert_size_stride(view_2, (512, 256), (256, 1))
    assert_size_stride(clone_default_33, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_34, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_35, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_204, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_205, (), ())
    assert_size_stride(getitem_206, (), ())
    assert_size_stride(alias_default_23, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_18, (512, 256), (256, 1))
    assert_size_stride(getitem_7, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_3, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_20, (512, 256), (256, 1))
    assert_size_stride(addmm_5, (512, 1024), (1024, 1))
    assert_size_stride(view_22, (512, 1024), (1024, 1))
    assert_size_stride(getitem_11, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_8, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_24, (512, 256), (256, 1))
    assert_size_stride(clone_default_30, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_31, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_32, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_197, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_198, (), ())
    assert_size_stride(getitem_199, (), ())
    assert_size_stride(alias_default_21, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_40, (512, 256), (256, 1))
    assert_size_stride(getitem_17, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_10, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_42, (512, 256), (256, 1))
    assert_size_stride(addmm_11, (512, 1024), (1024, 1))
    assert_size_stride(view_44, (512, 1024), (1024, 1))
    assert_size_stride(getitem_21, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_15, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_46, (512, 256), (256, 1))
    assert_size_stride(clone_default_27, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_28, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_29, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_190, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_191, (), ())
    assert_size_stride(getitem_192, (), ())
    assert_size_stride(alias_default_19, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_62, (512, 256), (256, 1))
    assert_size_stride(getitem_27, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_17, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_64, (512, 256), (256, 1))
    assert_size_stride(addmm_17, (512, 1024), (1024, 1))
    assert_size_stride(view_66, (512, 1024), (1024, 1))
    assert_size_stride(getitem_31, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_22, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_68, (512, 256), (256, 1))
    assert_size_stride(clone_default_24, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_25, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_26, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_183, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_184, (), ())
    assert_size_stride(getitem_185, (), ())
    assert_size_stride(alias_default_17, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_84, (512, 256), (256, 1))
    assert_size_stride(getitem_37, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_24, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_86, (512, 256), (256, 1))
    assert_size_stride(addmm_23, (512, 1024), (1024, 1))
    assert_size_stride(view_88, (512, 1024), (1024, 1))
    assert_size_stride(getitem_41, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_29, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_90, (512, 256), (256, 1))
    assert_size_stride(clone_default_21, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_22, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_23, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_176, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_177, (), ())
    assert_size_stride(getitem_178, (), ())
    assert_size_stride(alias_default_15, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_106, (512, 256), (256, 1))
    assert_size_stride(getitem_47, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_31, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_108, (512, 256), (256, 1))
    assert_size_stride(addmm_29, (512, 1024), (1024, 1))
    assert_size_stride(view_110, (512, 1024), (1024, 1))
    assert_size_stride(getitem_51, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_36, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_112, (512, 256), (256, 1))
    assert_size_stride(clone_default_18, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_19, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_20, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_169, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_170, (), ())
    assert_size_stride(getitem_171, (), ())
    assert_size_stride(alias_default_13, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_128, (512, 256), (256, 1))
    assert_size_stride(getitem_57, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_38, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_130, (512, 256), (256, 1))
    assert_size_stride(addmm_35, (512, 1024), (1024, 1))
    assert_size_stride(view_132, (512, 1024), (1024, 1))
    assert_size_stride(getitem_61, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_43, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_134, (512, 256), (256, 1))
    assert_size_stride(clone_default_15, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_16, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_17, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_162, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_163, (), ())
    assert_size_stride(getitem_164, (), ())
    assert_size_stride(alias_default_11, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_150, (512, 256), (256, 1))
    assert_size_stride(getitem_67, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_45, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_152, (512, 256), (256, 1))
    assert_size_stride(addmm_41, (512, 1024), (1024, 1))
    assert_size_stride(view_154, (512, 1024), (1024, 1))
    assert_size_stride(getitem_71, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_50, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_156, (512, 256), (256, 1))
    assert_size_stride(clone_default_12, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_13, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_14, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_155, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_156, (), ())
    assert_size_stride(getitem_157, (), ())
    assert_size_stride(alias_default_9, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_172, (512, 256), (256, 1))
    assert_size_stride(getitem_77, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_52, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_174, (512, 256), (256, 1))
    assert_size_stride(addmm_47, (512, 1024), (1024, 1))
    assert_size_stride(view_176, (512, 1024), (1024, 1))
    assert_size_stride(getitem_81, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_57, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_178, (512, 256), (256, 1))
    assert_size_stride(clone_default_9, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_10, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_11, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_148, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_149, (), ())
    assert_size_stride(getitem_150, (), ())
    assert_size_stride(alias_default_7, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_194, (512, 256), (256, 1))
    assert_size_stride(getitem_87, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_59, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_196, (512, 256), (256, 1))
    assert_size_stride(addmm_53, (512, 1024), (1024, 1))
    assert_size_stride(view_198, (512, 1024), (1024, 1))
    assert_size_stride(getitem_91, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_64, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_200, (512, 256), (256, 1))
    assert_size_stride(clone_default_6, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_7, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_8, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_141, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_142, (), ())
    assert_size_stride(getitem_143, (), ())
    assert_size_stride(alias_default_5, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_216, (512, 256), (256, 1))
    assert_size_stride(getitem_97, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_66, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_218, (512, 256), (256, 1))
    assert_size_stride(addmm_59, (512, 1024), (1024, 1))
    assert_size_stride(view_220, (512, 1024), (1024, 1))
    assert_size_stride(getitem_101, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_71, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_222, (512, 256), (256, 1))
    assert_size_stride(clone_default_3, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_4, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_5, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_134, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_135, (), ())
    assert_size_stride(getitem_136, (), ())
    assert_size_stride(alias_default_3, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_238, (512, 256), (256, 1))
    assert_size_stride(getitem_107, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_73, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_240, (512, 256), (256, 1))
    assert_size_stride(addmm_65, (512, 1024), (1024, 1))
    assert_size_stride(view_242, (512, 1024), (1024, 1))
    assert_size_stride(getitem_111, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_78, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_244, (512, 256), (256, 1))
    assert_size_stride(clone_default, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_1, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(clone_default_2, (1, 4, 512, 64), (131072, 32768, 64, 1))
    assert_size_stride(getitem_127, (1, 4, 512), (2048, 512, 1))
    assert_size_stride(getitem_128, (), ())
    assert_size_stride(getitem_129, (), ())
    assert_size_stride(alias_default_1, (1, 4, 512, 64), (131072, 64, 256, 1))
    assert_size_stride(view_260, (512, 256), (256, 1))
    assert_size_stride(getitem_117, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_80, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_262, (512, 256), (256, 1))
    assert_size_stride(addmm_71, (512, 1024), (1024, 1))
    assert_size_stride(view_264, (512, 1024), (1024, 1))
    assert_size_stride(getitem_121, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_85, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_266, (512, 256), (256, 1))
    assert_size_stride(sub_39, (1, 512), (512, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_41, (1, 512), (512, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_4, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_6, (1, 1), (1, 1))
    assert_size_stride(permute_134, (2, 256), (256, 1))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_138, (256, 1024), (1024, 1))
    assert_size_stride(permute_142, (1024, 256), (256, 1))
    assert_size_stride(div_31, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_146, (256, 256), (256, 1))
    assert_size_stride(permute_158, (256, 256), (256, 1))
    assert_size_stride(permute_163, (256, 256), (256, 1))
    assert_size_stride(permute_167, (256, 256), (256, 1))
    assert_size_stride(div_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_171, (256, 1024), (1024, 1))
    assert_size_stride(permute_175, (1024, 256), (256, 1))
    assert_size_stride(div_34, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_179, (256, 256), (256, 1))
    assert_size_stride(permute_191, (256, 256), (256, 1))
    assert_size_stride(permute_196, (256, 256), (256, 1))
    assert_size_stride(permute_200, (256, 256), (256, 1))
    assert_size_stride(div_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_204, (256, 1024), (1024, 1))
    assert_size_stride(permute_208, (1024, 256), (256, 1))
    assert_size_stride(div_37, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_212, (256, 256), (256, 1))
    assert_size_stride(permute_224, (256, 256), (256, 1))
    assert_size_stride(permute_229, (256, 256), (256, 1))
    assert_size_stride(permute_233, (256, 256), (256, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_237, (256, 1024), (1024, 1))
    assert_size_stride(permute_241, (1024, 256), (256, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_245, (256, 256), (256, 1))
    assert_size_stride(permute_257, (256, 256), (256, 1))
    assert_size_stride(permute_262, (256, 256), (256, 1))
    assert_size_stride(permute_266, (256, 256), (256, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_270, (256, 1024), (1024, 1))
    assert_size_stride(permute_274, (1024, 256), (256, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_278, (256, 256), (256, 1))
    assert_size_stride(permute_290, (256, 256), (256, 1))
    assert_size_stride(permute_295, (256, 256), (256, 1))
    assert_size_stride(permute_299, (256, 256), (256, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_303, (256, 1024), (1024, 1))
    assert_size_stride(permute_307, (1024, 256), (256, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_311, (256, 256), (256, 1))
    assert_size_stride(permute_323, (256, 256), (256, 1))
    assert_size_stride(permute_328, (256, 256), (256, 1))
    assert_size_stride(permute_332, (256, 256), (256, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_336, (256, 1024), (1024, 1))
    assert_size_stride(permute_340, (1024, 256), (256, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_344, (256, 256), (256, 1))
    assert_size_stride(permute_356, (256, 256), (256, 1))
    assert_size_stride(permute_361, (256, 256), (256, 1))
    assert_size_stride(permute_365, (256, 256), (256, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_369, (256, 1024), (1024, 1))
    assert_size_stride(permute_373, (1024, 256), (256, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_377, (256, 256), (256, 1))
    assert_size_stride(permute_389, (256, 256), (256, 1))
    assert_size_stride(permute_394, (256, 256), (256, 1))
    assert_size_stride(permute_398, (256, 256), (256, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_402, (256, 1024), (1024, 1))
    assert_size_stride(permute_406, (1024, 256), (256, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_410, (256, 256), (256, 1))
    assert_size_stride(permute_422, (256, 256), (256, 1))
    assert_size_stride(permute_427, (256, 256), (256, 1))
    assert_size_stride(permute_431, (256, 256), (256, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_435, (256, 1024), (1024, 1))
    assert_size_stride(permute_439, (1024, 256), (256, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_443, (256, 256), (256, 1))
    assert_size_stride(permute_455, (256, 256), (256, 1))
    assert_size_stride(permute_460, (256, 256), (256, 1))
    assert_size_stride(permute_464, (256, 256), (256, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_468, (256, 1024), (1024, 1))
    assert_size_stride(permute_472, (1024, 256), (256, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_476, (256, 256), (256, 1))
    assert_size_stride(permute_488, (256, 256), (256, 1))
    assert_size_stride(permute_493, (256, 256), (256, 1))
    assert_size_stride(permute_497, (256, 256), (256, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_501, (256, 1024), (1024, 1))
    assert_size_stride(permute_505, (1024, 256), (256, 1))
    assert_size_stride(div_64, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_509, (256, 256), (256, 1))
    assert_size_stride(permute_521, (256, 256), (256, 1))
    assert_size_stride(permute_526, (256, 256), (256, 1))
    assert_size_stride(permute_530, (256, 256), (256, 1))
    assert_size_stride(permute_534, (256, 128), (128, 1))
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
        buf9 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (512, 2), (2, 1), 0), permute_134, out=buf9)
        del permute_134
        buf10 = reinterpret_tensor(buf4, (2, 256), (256, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (2, 512), (1, 2), 0), view_266, out=buf10)
        del view_266
        buf11 = empty_strided((1, 2, 4), (8, 1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf8, buf11, 8, 128, grid=grid(8), stream=stream0)
        buf12 = empty((1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf11, buf12, 2, 4, grid=grid(2), stream=stream0)
        del buf11
        buf15 = empty((1, 512, 256), device='cuda', dtype=torch.float32)
        buf18 = empty((1, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_5.run(buf9, primals_198, mul_85, div_30, getitem_121, buf15, buf18, 512, 256, grid=grid(512), stream=stream0)
        del div_30
        del getitem_121
        del primals_198
        buf16 = empty((256, ), device='cuda', dtype=torch.float32)
        buf17 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf9, mul_85, buf16, buf17, 256, 512, grid=grid(256), stream=stream0)
        del mul_85
        buf19 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (512, 256), (256, 1), 0), permute_138, out=buf19)
        del permute_138
        buf20 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (256, 512), (1, 256), 0), view_264, out=buf20)
        del view_264
        buf21 = reinterpret_tensor(buf8, (1, 256, 4), (1024, 1, 256), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf18, buf21, 1024, 128, grid=grid(1024), stream=stream0)
        buf22 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf21, buf22, 256, 4, grid=grid(256), stream=stream0)
        buf23 = reinterpret_tensor(buf19, (1, 512, 1024), (524288, 1024, 1), 0); del buf19  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf23, addmm_71, 524288, grid=grid(524288), stream=stream0)
        del addmm_71
        buf24 = reinterpret_tensor(buf18, (512, 256), (256, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (512, 1024), (1024, 1), 0), permute_142, out=buf24)
        del permute_142
        buf25 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1024, 512), (1, 1024), 0), view_262, out=buf25)
        del view_262
        buf26 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf23, buf26, 4096, 128, grid=grid(4096), stream=stream0)
        buf27 = reinterpret_tensor(buf21, (1, 1024), (1024, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf26, buf27, 1024, 4, grid=grid(1024), stream=stream0)
        buf30 = reinterpret_tensor(buf9, (1, 512, 256), (131072, 256, 1), 0); del buf9  # reuse
        buf33 = empty((1, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf15, buf24, primals_192, mul_80, div_31, getitem_117, buf30, buf33, 512, 256, grid=grid(512), stream=stream0)
        del div_31
        del getitem_117
        del primals_192
        buf31 = empty((256, ), device='cuda', dtype=torch.float32)
        buf32 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf15, buf24, mul_80, buf31, buf32, 256, 512, grid=grid(256), stream=stream0)
        del buf15
        del mul_80
        buf34 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (512, 256), (256, 1), 0), permute_146, out=buf34)
        del permute_146
        buf35 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (256, 512), (1, 256), 0), view_260, out=buf35)
        del view_260
        buf36 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf33, buf36, 1024, 128, grid=grid(1024), stream=stream0)
        buf37 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf36, buf37, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf38 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf34, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_1
        del clone_default
        del clone_default_1
        del clone_default_2
        del getitem_127
        del getitem_128
        del getitem_129
        buf39 = buf38[0]
        buf40 = buf38[1]
        buf41 = buf38[2]
        del buf38
        buf42 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (512, 256), (256, 1), 0), permute_158, out=buf42)
        del permute_158
        buf43 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (256, 512), (1, 256), 0), view_244, out=buf43)
        buf44 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf41, buf44, 1024, 128, grid=grid(1024), stream=stream0)
        buf45 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf44, buf45, 256, 4, grid=grid(256), stream=stream0)
        buf46 = reinterpret_tensor(buf41, (512, 256), (256, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (512, 256), (256, 1), 0), permute_163, out=buf46)
        del permute_163
        buf47 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (256, 512), (1, 256), 0), view_244, out=buf47)
        buf48 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf40, buf48, 1024, 128, grid=grid(1024), stream=stream0)
        buf49 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf48, buf49, 256, 4, grid=grid(256), stream=stream0)
        buf50 = reinterpret_tensor(buf40, (512, 256), (256, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (512, 256), (256, 1), 0), permute_167, out=buf50)
        del permute_167
        buf51 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (256, 512), (1, 256), 0), view_244, out=buf51)
        del view_244
        buf52 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf39, buf52, 1024, 128, grid=grid(1024), stream=stream0)
        buf53 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf52, buf53, 256, 4, grid=grid(256), stream=stream0)
        buf57 = reinterpret_tensor(buf39, (1, 512, 256), (131072, 256, 1), 0); del buf39  # reuse
        buf60 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf30, buf42, buf46, buf50, primals_182, mul_78, div_33, getitem_111, buf57, buf60, 512, 256, grid=grid(512), stream=stream0)
        del div_33
        del getitem_111
        del primals_182
        buf58 = empty((256, ), device='cuda', dtype=torch.float32)
        buf59 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf30, buf42, buf46, buf50, mul_78, buf58, buf59, 256, 512, grid=grid(256), stream=stream0)
        del buf30
        del buf42
        del mul_78
        buf61 = reinterpret_tensor(buf23, (512, 1024), (1024, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (512, 256), (256, 1), 0), permute_171, out=buf61)
        del permute_171
        buf62 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (256, 512), (1, 256), 0), view_242, out=buf62)
        del view_242
        buf63 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf60, buf63, 1024, 128, grid=grid(1024), stream=stream0)
        buf64 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf63, buf64, 256, 4, grid=grid(256), stream=stream0)
        buf65 = reinterpret_tensor(buf61, (1, 512, 1024), (524288, 1024, 1), 0); del buf61  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf65, addmm_65, 524288, grid=grid(524288), stream=stream0)
        del addmm_65
        buf66 = reinterpret_tensor(buf60, (512, 256), (256, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (512, 1024), (1024, 1), 0), permute_175, out=buf66)
        del permute_175
        buf67 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (1024, 512), (1, 1024), 0), view_240, out=buf67)
        del view_240
        buf68 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf65, buf68, 4096, 128, grid=grid(4096), stream=stream0)
        buf69 = reinterpret_tensor(buf63, (1, 1024), (1024, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf68, buf69, 1024, 4, grid=grid(1024), stream=stream0)
        buf72 = reinterpret_tensor(buf50, (1, 512, 256), (131072, 256, 1), 0); del buf50  # reuse
        buf75 = reinterpret_tensor(buf46, (1, 512, 256), (131072, 256, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf57, buf66, primals_176, mul_73, div_34, getitem_107, buf72, buf75, 512, 256, grid=grid(512), stream=stream0)
        del div_34
        del getitem_107
        del primals_176
        buf73 = empty((256, ), device='cuda', dtype=torch.float32)
        buf74 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf57, buf66, mul_73, buf73, buf74, 256, 512, grid=grid(256), stream=stream0)
        del buf57
        del mul_73
        buf76 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (512, 256), (256, 1), 0), permute_179, out=buf76)
        del permute_179
        buf77 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (256, 512), (1, 256), 0), view_238, out=buf77)
        del view_238
        buf78 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf75, buf78, 1024, 128, grid=grid(1024), stream=stream0)
        buf79 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf78, buf79, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf80 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf76, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_134, getitem_135, getitem_136, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_3
        del clone_default_3
        del clone_default_4
        del clone_default_5
        del getitem_134
        del getitem_135
        del getitem_136
        buf81 = buf80[0]
        buf82 = buf80[1]
        buf83 = buf80[2]
        del buf80
        buf84 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (512, 256), (256, 1), 0), permute_191, out=buf84)
        del permute_191
        buf85 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (256, 512), (1, 256), 0), view_222, out=buf85)
        buf86 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf83, buf86, 1024, 128, grid=grid(1024), stream=stream0)
        buf87 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf86, buf87, 256, 4, grid=grid(256), stream=stream0)
        buf88 = reinterpret_tensor(buf83, (512, 256), (256, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (512, 256), (256, 1), 0), permute_196, out=buf88)
        del permute_196
        buf89 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (256, 512), (1, 256), 0), view_222, out=buf89)
        buf90 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf82, buf90, 1024, 128, grid=grid(1024), stream=stream0)
        buf91 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf90, buf91, 256, 4, grid=grid(256), stream=stream0)
        buf92 = reinterpret_tensor(buf82, (512, 256), (256, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (512, 256), (256, 1), 0), permute_200, out=buf92)
        del permute_200
        buf93 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (256, 512), (1, 256), 0), view_222, out=buf93)
        del view_222
        buf94 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf81, buf94, 1024, 128, grid=grid(1024), stream=stream0)
        buf95 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf94, buf95, 256, 4, grid=grid(256), stream=stream0)
        buf99 = reinterpret_tensor(buf81, (1, 512, 256), (131072, 256, 1), 0); del buf81  # reuse
        buf102 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf72, buf84, buf88, buf92, primals_166, mul_71, div_36, getitem_101, buf99, buf102, 512, 256, grid=grid(512), stream=stream0)
        del div_36
        del getitem_101
        del primals_166
        buf100 = empty((256, ), device='cuda', dtype=torch.float32)
        buf101 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf72, buf84, buf88, buf92, mul_71, buf100, buf101, 256, 512, grid=grid(256), stream=stream0)
        del buf72
        del buf84
        del mul_71
        buf103 = reinterpret_tensor(buf65, (512, 1024), (1024, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (512, 256), (256, 1), 0), permute_204, out=buf103)
        del permute_204
        buf104 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (256, 512), (1, 256), 0), view_220, out=buf104)
        del view_220
        buf105 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf102, buf105, 1024, 128, grid=grid(1024), stream=stream0)
        buf106 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf105, buf106, 256, 4, grid=grid(256), stream=stream0)
        buf107 = reinterpret_tensor(buf103, (1, 512, 1024), (524288, 1024, 1), 0); del buf103  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf107, addmm_59, 524288, grid=grid(524288), stream=stream0)
        del addmm_59
        buf108 = reinterpret_tensor(buf102, (512, 256), (256, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (512, 1024), (1024, 1), 0), permute_208, out=buf108)
        del permute_208
        buf109 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (1024, 512), (1, 1024), 0), view_218, out=buf109)
        del view_218
        buf110 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf107, buf110, 4096, 128, grid=grid(4096), stream=stream0)
        buf111 = reinterpret_tensor(buf105, (1, 1024), (1024, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf110, buf111, 1024, 4, grid=grid(1024), stream=stream0)
        buf114 = reinterpret_tensor(buf92, (1, 512, 256), (131072, 256, 1), 0); del buf92  # reuse
        buf117 = reinterpret_tensor(buf88, (1, 512, 256), (131072, 256, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf99, buf108, primals_160, mul_66, div_37, getitem_97, buf114, buf117, 512, 256, grid=grid(512), stream=stream0)
        del div_37
        del getitem_97
        del primals_160
        buf115 = empty((256, ), device='cuda', dtype=torch.float32)
        buf116 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf99, buf108, mul_66, buf115, buf116, 256, 512, grid=grid(256), stream=stream0)
        del buf108
        del mul_66
        buf118 = reinterpret_tensor(buf99, (512, 256), (256, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (512, 256), (256, 1), 0), permute_212, out=buf118)
        del permute_212
        buf119 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (256, 512), (1, 256), 0), view_216, out=buf119)
        del view_216
        buf120 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf117, buf120, 1024, 128, grid=grid(1024), stream=stream0)
        buf121 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf120, buf121, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf122 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf118, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_141, getitem_142, getitem_143, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_5
        del clone_default_6
        del clone_default_7
        del clone_default_8
        del getitem_141
        del getitem_142
        del getitem_143
        buf123 = buf122[0]
        buf124 = buf122[1]
        buf125 = buf122[2]
        del buf122
        buf126 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (512, 256), (256, 1), 0), permute_224, out=buf126)
        del permute_224
        buf127 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (256, 512), (1, 256), 0), view_200, out=buf127)
        buf128 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf125, buf128, 1024, 128, grid=grid(1024), stream=stream0)
        buf129 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf128, buf129, 256, 4, grid=grid(256), stream=stream0)
        buf130 = reinterpret_tensor(buf125, (512, 256), (256, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (512, 256), (256, 1), 0), permute_229, out=buf130)
        del permute_229
        buf131 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (256, 512), (1, 256), 0), view_200, out=buf131)
        buf132 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf124, buf132, 1024, 128, grid=grid(1024), stream=stream0)
        buf133 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf132, buf133, 256, 4, grid=grid(256), stream=stream0)
        buf134 = reinterpret_tensor(buf124, (512, 256), (256, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (512, 256), (256, 1), 0), permute_233, out=buf134)
        del permute_233
        buf135 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (256, 512), (1, 256), 0), view_200, out=buf135)
        del view_200
        buf136 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf123, buf136, 1024, 128, grid=grid(1024), stream=stream0)
        buf137 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf136, buf137, 256, 4, grid=grid(256), stream=stream0)
        buf141 = reinterpret_tensor(buf123, (1, 512, 256), (131072, 256, 1), 0); del buf123  # reuse
        buf144 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf114, buf126, buf130, buf134, primals_150, mul_64, div_39, getitem_91, buf141, buf144, 512, 256, grid=grid(512), stream=stream0)
        del div_39
        del getitem_91
        del primals_150
        buf142 = empty((256, ), device='cuda', dtype=torch.float32)
        buf143 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf114, buf126, buf130, buf134, mul_64, buf142, buf143, 256, 512, grid=grid(256), stream=stream0)
        del buf114
        del buf126
        del mul_64
        buf145 = reinterpret_tensor(buf107, (512, 1024), (1024, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (512, 256), (256, 1), 0), permute_237, out=buf145)
        del permute_237
        buf146 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (256, 512), (1, 256), 0), view_198, out=buf146)
        del view_198
        buf147 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf144, buf147, 1024, 128, grid=grid(1024), stream=stream0)
        buf148 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf147, buf148, 256, 4, grid=grid(256), stream=stream0)
        buf149 = reinterpret_tensor(buf145, (1, 512, 1024), (524288, 1024, 1), 0); del buf145  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf149, addmm_53, 524288, grid=grid(524288), stream=stream0)
        del addmm_53
        buf150 = reinterpret_tensor(buf144, (512, 256), (256, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (512, 1024), (1024, 1), 0), permute_241, out=buf150)
        del permute_241
        buf151 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (1024, 512), (1, 1024), 0), view_196, out=buf151)
        del view_196
        buf152 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf149, buf152, 4096, 128, grid=grid(4096), stream=stream0)
        buf153 = reinterpret_tensor(buf147, (1, 1024), (1024, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf152, buf153, 1024, 4, grid=grid(1024), stream=stream0)
        buf156 = reinterpret_tensor(buf134, (1, 512, 256), (131072, 256, 1), 0); del buf134  # reuse
        buf159 = reinterpret_tensor(buf130, (1, 512, 256), (131072, 256, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf141, buf150, primals_144, mul_59, div_40, getitem_87, buf156, buf159, 512, 256, grid=grid(512), stream=stream0)
        del div_40
        del getitem_87
        del primals_144
        buf157 = empty((256, ), device='cuda', dtype=torch.float32)
        buf158 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf141, buf150, mul_59, buf157, buf158, 256, 512, grid=grid(256), stream=stream0)
        del buf141
        del mul_59
        buf160 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (512, 256), (256, 1), 0), permute_245, out=buf160)
        del permute_245
        buf161 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (256, 512), (1, 256), 0), view_194, out=buf161)
        del view_194
        buf162 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf159, buf162, 1024, 128, grid=grid(1024), stream=stream0)
        buf163 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf162, buf163, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf164 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf160, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_148, getitem_149, getitem_150, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_7
        del clone_default_10
        del clone_default_11
        del clone_default_9
        del getitem_148
        del getitem_149
        del getitem_150
        buf165 = buf164[0]
        buf166 = buf164[1]
        buf167 = buf164[2]
        del buf164
        buf168 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (512, 256), (256, 1), 0), permute_257, out=buf168)
        del permute_257
        buf169 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (256, 512), (1, 256), 0), view_178, out=buf169)
        buf170 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf167, buf170, 1024, 128, grid=grid(1024), stream=stream0)
        buf171 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf170, buf171, 256, 4, grid=grid(256), stream=stream0)
        buf172 = reinterpret_tensor(buf167, (512, 256), (256, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 256), (256, 1), 0), permute_262, out=buf172)
        del permute_262
        buf173 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (256, 512), (1, 256), 0), view_178, out=buf173)
        buf174 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf166, buf174, 1024, 128, grid=grid(1024), stream=stream0)
        buf175 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf174, buf175, 256, 4, grid=grid(256), stream=stream0)
        buf176 = reinterpret_tensor(buf166, (512, 256), (256, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (512, 256), (256, 1), 0), permute_266, out=buf176)
        del permute_266
        buf177 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (256, 512), (1, 256), 0), view_178, out=buf177)
        del view_178
        buf178 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf165, buf178, 1024, 128, grid=grid(1024), stream=stream0)
        buf179 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf178, buf179, 256, 4, grid=grid(256), stream=stream0)
        buf183 = reinterpret_tensor(buf165, (1, 512, 256), (131072, 256, 1), 0); del buf165  # reuse
        buf186 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf156, buf168, buf172, buf176, primals_134, mul_57, div_42, getitem_81, buf183, buf186, 512, 256, grid=grid(512), stream=stream0)
        del div_42
        del getitem_81
        del primals_134
        buf184 = empty((256, ), device='cuda', dtype=torch.float32)
        buf185 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf156, buf168, buf172, buf176, mul_57, buf184, buf185, 256, 512, grid=grid(256), stream=stream0)
        del buf156
        del buf168
        del mul_57
        buf187 = reinterpret_tensor(buf149, (512, 1024), (1024, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (512, 256), (256, 1), 0), permute_270, out=buf187)
        del permute_270
        buf188 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (256, 512), (1, 256), 0), view_176, out=buf188)
        del view_176
        buf189 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf186, buf189, 1024, 128, grid=grid(1024), stream=stream0)
        buf190 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf189, buf190, 256, 4, grid=grid(256), stream=stream0)
        buf191 = reinterpret_tensor(buf187, (1, 512, 1024), (524288, 1024, 1), 0); del buf187  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf191, addmm_47, 524288, grid=grid(524288), stream=stream0)
        del addmm_47
        buf192 = reinterpret_tensor(buf186, (512, 256), (256, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (512, 1024), (1024, 1), 0), permute_274, out=buf192)
        del permute_274
        buf193 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (1024, 512), (1, 1024), 0), view_174, out=buf193)
        del view_174
        buf194 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf191, buf194, 4096, 128, grid=grid(4096), stream=stream0)
        buf195 = reinterpret_tensor(buf189, (1, 1024), (1024, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf194, buf195, 1024, 4, grid=grid(1024), stream=stream0)
        buf198 = reinterpret_tensor(buf176, (1, 512, 256), (131072, 256, 1), 0); del buf176  # reuse
        buf201 = reinterpret_tensor(buf172, (1, 512, 256), (131072, 256, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf183, buf192, primals_128, mul_52, div_43, getitem_77, buf198, buf201, 512, 256, grid=grid(512), stream=stream0)
        del div_43
        del getitem_77
        del primals_128
        buf199 = empty((256, ), device='cuda', dtype=torch.float32)
        buf200 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf183, buf192, mul_52, buf199, buf200, 256, 512, grid=grid(256), stream=stream0)
        del buf183
        del mul_52
        buf202 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (512, 256), (256, 1), 0), permute_278, out=buf202)
        del permute_278
        buf203 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (256, 512), (1, 256), 0), view_172, out=buf203)
        del view_172
        buf204 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf201, buf204, 1024, 128, grid=grid(1024), stream=stream0)
        buf205 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf204, buf205, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf206 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf202, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_155, getitem_156, getitem_157, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_9
        del clone_default_12
        del clone_default_13
        del clone_default_14
        del getitem_155
        del getitem_156
        del getitem_157
        buf207 = buf206[0]
        buf208 = buf206[1]
        buf209 = buf206[2]
        del buf206
        buf210 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (512, 256), (256, 1), 0), permute_290, out=buf210)
        del permute_290
        buf211 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (256, 512), (1, 256), 0), view_156, out=buf211)
        buf212 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf209, buf212, 1024, 128, grid=grid(1024), stream=stream0)
        buf213 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf212, buf213, 256, 4, grid=grid(256), stream=stream0)
        buf214 = reinterpret_tensor(buf209, (512, 256), (256, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (512, 256), (256, 1), 0), permute_295, out=buf214)
        del permute_295
        buf215 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (256, 512), (1, 256), 0), view_156, out=buf215)
        buf216 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf208, buf216, 1024, 128, grid=grid(1024), stream=stream0)
        buf217 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf216, buf217, 256, 4, grid=grid(256), stream=stream0)
        buf218 = reinterpret_tensor(buf208, (512, 256), (256, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (512, 256), (256, 1), 0), permute_299, out=buf218)
        del permute_299
        buf219 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (256, 512), (1, 256), 0), view_156, out=buf219)
        del view_156
        buf220 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf207, buf220, 1024, 128, grid=grid(1024), stream=stream0)
        buf221 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf220, buf221, 256, 4, grid=grid(256), stream=stream0)
        buf225 = reinterpret_tensor(buf207, (1, 512, 256), (131072, 256, 1), 0); del buf207  # reuse
        buf228 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf198, buf210, buf214, buf218, primals_118, mul_50, div_45, getitem_71, buf225, buf228, 512, 256, grid=grid(512), stream=stream0)
        del div_45
        del getitem_71
        del primals_118
        buf226 = empty((256, ), device='cuda', dtype=torch.float32)
        buf227 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf198, buf210, buf214, buf218, mul_50, buf226, buf227, 256, 512, grid=grid(256), stream=stream0)
        del buf198
        del buf210
        del mul_50
        buf229 = reinterpret_tensor(buf191, (512, 1024), (1024, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (512, 256), (256, 1), 0), permute_303, out=buf229)
        del permute_303
        buf230 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (256, 512), (1, 256), 0), view_154, out=buf230)
        del view_154
        buf231 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf228, buf231, 1024, 128, grid=grid(1024), stream=stream0)
        buf232 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf231, buf232, 256, 4, grid=grid(256), stream=stream0)
        buf233 = reinterpret_tensor(buf229, (1, 512, 1024), (524288, 1024, 1), 0); del buf229  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf233, addmm_41, 524288, grid=grid(524288), stream=stream0)
        del addmm_41
        buf234 = reinterpret_tensor(buf228, (512, 256), (256, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (512, 1024), (1024, 1), 0), permute_307, out=buf234)
        del permute_307
        buf235 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (1024, 512), (1, 1024), 0), view_152, out=buf235)
        del view_152
        buf236 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf233, buf236, 4096, 128, grid=grid(4096), stream=stream0)
        buf237 = reinterpret_tensor(buf231, (1, 1024), (1024, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf236, buf237, 1024, 4, grid=grid(1024), stream=stream0)
        buf240 = reinterpret_tensor(buf218, (1, 512, 256), (131072, 256, 1), 0); del buf218  # reuse
        buf243 = reinterpret_tensor(buf214, (1, 512, 256), (131072, 256, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf225, buf234, primals_112, mul_45, div_46, getitem_67, buf240, buf243, 512, 256, grid=grid(512), stream=stream0)
        del div_46
        del getitem_67
        del primals_112
        buf241 = empty((256, ), device='cuda', dtype=torch.float32)
        buf242 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf225, buf234, mul_45, buf241, buf242, 256, 512, grid=grid(256), stream=stream0)
        del buf225
        del mul_45
        buf244 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (512, 256), (256, 1), 0), permute_311, out=buf244)
        del permute_311
        buf245 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (256, 512), (1, 256), 0), view_150, out=buf245)
        del view_150
        buf246 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf243, buf246, 1024, 128, grid=grid(1024), stream=stream0)
        buf247 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf246, buf247, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf248 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf244, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_162, getitem_163, getitem_164, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_11
        del clone_default_15
        del clone_default_16
        del clone_default_17
        del getitem_162
        del getitem_163
        del getitem_164
        buf249 = buf248[0]
        buf250 = buf248[1]
        buf251 = buf248[2]
        del buf248
        buf252 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (512, 256), (256, 1), 0), permute_323, out=buf252)
        del permute_323
        buf253 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (256, 512), (1, 256), 0), view_134, out=buf253)
        buf254 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf251, buf254, 1024, 128, grid=grid(1024), stream=stream0)
        buf255 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf254, buf255, 256, 4, grid=grid(256), stream=stream0)
        buf256 = reinterpret_tensor(buf251, (512, 256), (256, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (512, 256), (256, 1), 0), permute_328, out=buf256)
        del permute_328
        buf257 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (256, 512), (1, 256), 0), view_134, out=buf257)
        buf258 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf250, buf258, 1024, 128, grid=grid(1024), stream=stream0)
        buf259 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf258, buf259, 256, 4, grid=grid(256), stream=stream0)
        buf260 = reinterpret_tensor(buf250, (512, 256), (256, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 256), (256, 1), 0), permute_332, out=buf260)
        del permute_332
        buf261 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (256, 512), (1, 256), 0), view_134, out=buf261)
        del view_134
        buf262 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf249, buf262, 1024, 128, grid=grid(1024), stream=stream0)
        buf263 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf262, buf263, 256, 4, grid=grid(256), stream=stream0)
        buf267 = reinterpret_tensor(buf249, (1, 512, 256), (131072, 256, 1), 0); del buf249  # reuse
        buf270 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf240, buf252, buf256, buf260, primals_102, mul_43, div_48, getitem_61, buf267, buf270, 512, 256, grid=grid(512), stream=stream0)
        del div_48
        del getitem_61
        del primals_102
        buf268 = empty((256, ), device='cuda', dtype=torch.float32)
        buf269 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf240, buf252, buf256, buf260, mul_43, buf268, buf269, 256, 512, grid=grid(256), stream=stream0)
        del buf240
        del buf252
        del mul_43
        buf271 = reinterpret_tensor(buf233, (512, 1024), (1024, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (512, 256), (256, 1), 0), permute_336, out=buf271)
        del permute_336
        buf272 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (256, 512), (1, 256), 0), view_132, out=buf272)
        del view_132
        buf273 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf270, buf273, 1024, 128, grid=grid(1024), stream=stream0)
        buf274 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf273, buf274, 256, 4, grid=grid(256), stream=stream0)
        buf275 = reinterpret_tensor(buf271, (1, 512, 1024), (524288, 1024, 1), 0); del buf271  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf275, addmm_35, 524288, grid=grid(524288), stream=stream0)
        del addmm_35
        buf276 = reinterpret_tensor(buf270, (512, 256), (256, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (512, 1024), (1024, 1), 0), permute_340, out=buf276)
        del permute_340
        buf277 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (1024, 512), (1, 1024), 0), view_130, out=buf277)
        del view_130
        buf278 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf275, buf278, 4096, 128, grid=grid(4096), stream=stream0)
        buf279 = reinterpret_tensor(buf273, (1, 1024), (1024, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf278, buf279, 1024, 4, grid=grid(1024), stream=stream0)
        buf282 = reinterpret_tensor(buf260, (1, 512, 256), (131072, 256, 1), 0); del buf260  # reuse
        buf285 = reinterpret_tensor(buf256, (1, 512, 256), (131072, 256, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf267, buf276, primals_96, mul_38, div_49, getitem_57, buf282, buf285, 512, 256, grid=grid(512), stream=stream0)
        del div_49
        del getitem_57
        del primals_96
        buf283 = empty((256, ), device='cuda', dtype=torch.float32)
        buf284 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf267, buf276, mul_38, buf283, buf284, 256, 512, grid=grid(256), stream=stream0)
        del buf267
        del mul_38
        buf286 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (512, 256), (256, 1), 0), permute_344, out=buf286)
        del permute_344
        buf287 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (256, 512), (1, 256), 0), view_128, out=buf287)
        del view_128
        buf288 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf285, buf288, 1024, 128, grid=grid(1024), stream=stream0)
        buf289 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf288, buf289, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf290 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf286, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_169, getitem_170, getitem_171, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_13
        del clone_default_18
        del clone_default_19
        del clone_default_20
        del getitem_169
        del getitem_170
        del getitem_171
        buf291 = buf290[0]
        buf292 = buf290[1]
        buf293 = buf290[2]
        del buf290
        buf294 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (512, 256), (256, 1), 0), permute_356, out=buf294)
        del permute_356
        buf295 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (256, 512), (1, 256), 0), view_112, out=buf295)
        buf296 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf293, buf296, 1024, 128, grid=grid(1024), stream=stream0)
        buf297 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf296, buf297, 256, 4, grid=grid(256), stream=stream0)
        buf298 = reinterpret_tensor(buf293, (512, 256), (256, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (512, 256), (256, 1), 0), permute_361, out=buf298)
        del permute_361
        buf299 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (256, 512), (1, 256), 0), view_112, out=buf299)
        buf300 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf292, buf300, 1024, 128, grid=grid(1024), stream=stream0)
        buf301 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf300, buf301, 256, 4, grid=grid(256), stream=stream0)
        buf302 = reinterpret_tensor(buf292, (512, 256), (256, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (512, 256), (256, 1), 0), permute_365, out=buf302)
        del permute_365
        buf303 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (256, 512), (1, 256), 0), view_112, out=buf303)
        del view_112
        buf304 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf291, buf304, 1024, 128, grid=grid(1024), stream=stream0)
        buf305 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf304, buf305, 256, 4, grid=grid(256), stream=stream0)
        buf309 = reinterpret_tensor(buf291, (1, 512, 256), (131072, 256, 1), 0); del buf291  # reuse
        buf312 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf282, buf294, buf298, buf302, primals_86, mul_36, div_51, getitem_51, buf309, buf312, 512, 256, grid=grid(512), stream=stream0)
        del div_51
        del getitem_51
        del primals_86
        buf310 = empty((256, ), device='cuda', dtype=torch.float32)
        buf311 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf282, buf294, buf298, buf302, mul_36, buf310, buf311, 256, 512, grid=grid(256), stream=stream0)
        del buf282
        del buf294
        del mul_36
        buf313 = reinterpret_tensor(buf275, (512, 1024), (1024, 1), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (512, 256), (256, 1), 0), permute_369, out=buf313)
        del permute_369
        buf314 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (256, 512), (1, 256), 0), view_110, out=buf314)
        del view_110
        buf315 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf312, buf315, 1024, 128, grid=grid(1024), stream=stream0)
        buf316 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf315, buf316, 256, 4, grid=grid(256), stream=stream0)
        buf317 = reinterpret_tensor(buf313, (1, 512, 1024), (524288, 1024, 1), 0); del buf313  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf317, addmm_29, 524288, grid=grid(524288), stream=stream0)
        del addmm_29
        buf318 = reinterpret_tensor(buf312, (512, 256), (256, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (512, 1024), (1024, 1), 0), permute_373, out=buf318)
        del permute_373
        buf319 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (1024, 512), (1, 1024), 0), view_108, out=buf319)
        del view_108
        buf320 = buf278; del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf317, buf320, 4096, 128, grid=grid(4096), stream=stream0)
        buf321 = reinterpret_tensor(buf315, (1, 1024), (1024, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf320, buf321, 1024, 4, grid=grid(1024), stream=stream0)
        buf324 = reinterpret_tensor(buf302, (1, 512, 256), (131072, 256, 1), 0); del buf302  # reuse
        buf327 = reinterpret_tensor(buf298, (1, 512, 256), (131072, 256, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf309, buf318, primals_80, mul_31, div_52, getitem_47, buf324, buf327, 512, 256, grid=grid(512), stream=stream0)
        del div_52
        del getitem_47
        del primals_80
        buf325 = empty((256, ), device='cuda', dtype=torch.float32)
        buf326 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf309, buf318, mul_31, buf325, buf326, 256, 512, grid=grid(256), stream=stream0)
        del buf309
        del mul_31
        buf328 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (512, 256), (256, 1), 0), permute_377, out=buf328)
        del permute_377
        buf329 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (256, 512), (1, 256), 0), view_106, out=buf329)
        del view_106
        buf330 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf327, buf330, 1024, 128, grid=grid(1024), stream=stream0)
        buf331 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf330, buf331, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf332 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf328, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_176, getitem_177, getitem_178, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_15
        del clone_default_21
        del clone_default_22
        del clone_default_23
        del getitem_176
        del getitem_177
        del getitem_178
        buf333 = buf332[0]
        buf334 = buf332[1]
        buf335 = buf332[2]
        del buf332
        buf336 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (512, 256), (256, 1), 0), permute_389, out=buf336)
        del permute_389
        buf337 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (256, 512), (1, 256), 0), view_90, out=buf337)
        buf338 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf335, buf338, 1024, 128, grid=grid(1024), stream=stream0)
        buf339 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf338, buf339, 256, 4, grid=grid(256), stream=stream0)
        buf340 = reinterpret_tensor(buf335, (512, 256), (256, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (512, 256), (256, 1), 0), permute_394, out=buf340)
        del permute_394
        buf341 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (256, 512), (1, 256), 0), view_90, out=buf341)
        buf342 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf334, buf342, 1024, 128, grid=grid(1024), stream=stream0)
        buf343 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf342, buf343, 256, 4, grid=grid(256), stream=stream0)
        buf344 = reinterpret_tensor(buf334, (512, 256), (256, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (512, 256), (256, 1), 0), permute_398, out=buf344)
        del permute_398
        buf345 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (256, 512), (1, 256), 0), view_90, out=buf345)
        del view_90
        buf346 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf333, buf346, 1024, 128, grid=grid(1024), stream=stream0)
        buf347 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf346, buf347, 256, 4, grid=grid(256), stream=stream0)
        buf351 = reinterpret_tensor(buf333, (1, 512, 256), (131072, 256, 1), 0); del buf333  # reuse
        buf354 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf324, buf336, buf340, buf344, primals_70, mul_29, div_54, getitem_41, buf351, buf354, 512, 256, grid=grid(512), stream=stream0)
        del div_54
        del getitem_41
        del primals_70
        buf352 = empty((256, ), device='cuda', dtype=torch.float32)
        buf353 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf324, buf336, buf340, buf344, mul_29, buf352, buf353, 256, 512, grid=grid(256), stream=stream0)
        del buf324
        del buf336
        del mul_29
        buf355 = reinterpret_tensor(buf317, (512, 1024), (1024, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf354, (512, 256), (256, 1), 0), permute_402, out=buf355)
        del permute_402
        buf356 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf354, (256, 512), (1, 256), 0), view_88, out=buf356)
        del view_88
        buf357 = buf346; del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf354, buf357, 1024, 128, grid=grid(1024), stream=stream0)
        buf358 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf357, buf358, 256, 4, grid=grid(256), stream=stream0)
        buf359 = reinterpret_tensor(buf355, (1, 512, 1024), (524288, 1024, 1), 0); del buf355  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf359, addmm_23, 524288, grid=grid(524288), stream=stream0)
        del addmm_23
        buf360 = reinterpret_tensor(buf354, (512, 256), (256, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf359, (512, 1024), (1024, 1), 0), permute_406, out=buf360)
        del permute_406
        buf361 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf359, (1024, 512), (1, 1024), 0), view_86, out=buf361)
        del view_86
        buf362 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf359, buf362, 4096, 128, grid=grid(4096), stream=stream0)
        buf363 = reinterpret_tensor(buf357, (1, 1024), (1024, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf362, buf363, 1024, 4, grid=grid(1024), stream=stream0)
        buf366 = reinterpret_tensor(buf344, (1, 512, 256), (131072, 256, 1), 0); del buf344  # reuse
        buf369 = reinterpret_tensor(buf340, (1, 512, 256), (131072, 256, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf351, buf360, primals_64, mul_24, div_55, getitem_37, buf366, buf369, 512, 256, grid=grid(512), stream=stream0)
        del div_55
        del getitem_37
        del primals_64
        buf367 = empty((256, ), device='cuda', dtype=torch.float32)
        buf368 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf351, buf360, mul_24, buf367, buf368, 256, 512, grid=grid(256), stream=stream0)
        del buf351
        del mul_24
        buf370 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (512, 256), (256, 1), 0), permute_410, out=buf370)
        del permute_410
        buf371 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (256, 512), (1, 256), 0), view_84, out=buf371)
        del view_84
        buf372 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf369, buf372, 1024, 128, grid=grid(1024), stream=stream0)
        buf373 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf372, buf373, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf374 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf370, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_183, getitem_184, getitem_185, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_17
        del clone_default_24
        del clone_default_25
        del clone_default_26
        del getitem_183
        del getitem_184
        del getitem_185
        buf375 = buf374[0]
        buf376 = buf374[1]
        buf377 = buf374[2]
        del buf374
        buf378 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf377, (512, 256), (256, 1), 0), permute_422, out=buf378)
        del permute_422
        buf379 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf377, (256, 512), (1, 256), 0), view_68, out=buf379)
        buf380 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf377, buf380, 1024, 128, grid=grid(1024), stream=stream0)
        buf381 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf380, buf381, 256, 4, grid=grid(256), stream=stream0)
        buf382 = reinterpret_tensor(buf377, (512, 256), (256, 1), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (512, 256), (256, 1), 0), permute_427, out=buf382)
        del permute_427
        buf383 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (256, 512), (1, 256), 0), view_68, out=buf383)
        buf384 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf376, buf384, 1024, 128, grid=grid(1024), stream=stream0)
        buf385 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf384, buf385, 256, 4, grid=grid(256), stream=stream0)
        buf386 = reinterpret_tensor(buf376, (512, 256), (256, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (512, 256), (256, 1), 0), permute_431, out=buf386)
        del permute_431
        buf387 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (256, 512), (1, 256), 0), view_68, out=buf387)
        del view_68
        buf388 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf375, buf388, 1024, 128, grid=grid(1024), stream=stream0)
        buf389 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf388, buf389, 256, 4, grid=grid(256), stream=stream0)
        buf393 = reinterpret_tensor(buf375, (1, 512, 256), (131072, 256, 1), 0); del buf375  # reuse
        buf396 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf366, buf378, buf382, buf386, primals_54, mul_22, div_57, getitem_31, buf393, buf396, 512, 256, grid=grid(512), stream=stream0)
        del div_57
        del getitem_31
        del primals_54
        buf394 = empty((256, ), device='cuda', dtype=torch.float32)
        buf395 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf366, buf378, buf382, buf386, mul_22, buf394, buf395, 256, 512, grid=grid(256), stream=stream0)
        del buf366
        del buf378
        del mul_22
        buf397 = reinterpret_tensor(buf359, (512, 1024), (1024, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf396, (512, 256), (256, 1), 0), permute_435, out=buf397)
        del permute_435
        buf398 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf396, (256, 512), (1, 256), 0), view_66, out=buf398)
        del view_66
        buf399 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf396, buf399, 1024, 128, grid=grid(1024), stream=stream0)
        buf400 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf399, buf400, 256, 4, grid=grid(256), stream=stream0)
        buf401 = reinterpret_tensor(buf397, (1, 512, 1024), (524288, 1024, 1), 0); del buf397  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf401, addmm_17, 524288, grid=grid(524288), stream=stream0)
        del addmm_17
        buf402 = reinterpret_tensor(buf396, (512, 256), (256, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (512, 1024), (1024, 1), 0), permute_439, out=buf402)
        del permute_439
        buf403 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (1024, 512), (1, 1024), 0), view_64, out=buf403)
        del view_64
        buf404 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf401, buf404, 4096, 128, grid=grid(4096), stream=stream0)
        buf405 = reinterpret_tensor(buf399, (1, 1024), (1024, 1), 0); del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf404, buf405, 1024, 4, grid=grid(1024), stream=stream0)
        buf408 = reinterpret_tensor(buf386, (1, 512, 256), (131072, 256, 1), 0); del buf386  # reuse
        buf411 = reinterpret_tensor(buf382, (1, 512, 256), (131072, 256, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf393, buf402, primals_48, mul_17, div_58, getitem_27, buf408, buf411, 512, 256, grid=grid(512), stream=stream0)
        del div_58
        del getitem_27
        del primals_48
        buf409 = empty((256, ), device='cuda', dtype=torch.float32)
        buf410 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf393, buf402, mul_17, buf409, buf410, 256, 512, grid=grid(256), stream=stream0)
        del buf393
        del mul_17
        buf412 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (512, 256), (256, 1), 0), permute_443, out=buf412)
        del permute_443
        buf413 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (256, 512), (1, 256), 0), view_62, out=buf413)
        del view_62
        buf414 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf411, buf414, 1024, 128, grid=grid(1024), stream=stream0)
        buf415 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf414, buf415, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf416 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf412, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_190, getitem_191, getitem_192, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_19
        del clone_default_27
        del clone_default_28
        del clone_default_29
        del getitem_190
        del getitem_191
        del getitem_192
        buf417 = buf416[0]
        buf418 = buf416[1]
        buf419 = buf416[2]
        del buf416
        buf420 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (512, 256), (256, 1), 0), permute_455, out=buf420)
        del permute_455
        buf421 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (256, 512), (1, 256), 0), view_46, out=buf421)
        buf422 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf419, buf422, 1024, 128, grid=grid(1024), stream=stream0)
        buf423 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf422, buf423, 256, 4, grid=grid(256), stream=stream0)
        buf424 = reinterpret_tensor(buf419, (512, 256), (256, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf418, (512, 256), (256, 1), 0), permute_460, out=buf424)
        del permute_460
        buf425 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf418, (256, 512), (1, 256), 0), view_46, out=buf425)
        buf426 = buf422; del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf418, buf426, 1024, 128, grid=grid(1024), stream=stream0)
        buf427 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf426, buf427, 256, 4, grid=grid(256), stream=stream0)
        buf428 = reinterpret_tensor(buf418, (512, 256), (256, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (512, 256), (256, 1), 0), permute_464, out=buf428)
        del permute_464
        buf429 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (256, 512), (1, 256), 0), view_46, out=buf429)
        del view_46
        buf430 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf417, buf430, 1024, 128, grid=grid(1024), stream=stream0)
        buf431 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf430, buf431, 256, 4, grid=grid(256), stream=stream0)
        buf435 = reinterpret_tensor(buf417, (1, 512, 256), (131072, 256, 1), 0); del buf417  # reuse
        buf438 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf408, buf420, buf424, buf428, primals_38, mul_15, div_60, getitem_21, buf435, buf438, 512, 256, grid=grid(512), stream=stream0)
        del div_60
        del getitem_21
        del primals_38
        buf436 = empty((256, ), device='cuda', dtype=torch.float32)
        buf437 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf408, buf420, buf424, buf428, mul_15, buf436, buf437, 256, 512, grid=grid(256), stream=stream0)
        del buf408
        del buf420
        del mul_15
        buf439 = reinterpret_tensor(buf401, (512, 1024), (1024, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf438, (512, 256), (256, 1), 0), permute_468, out=buf439)
        del permute_468
        buf440 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf438, (256, 512), (1, 256), 0), view_44, out=buf440)
        del view_44
        buf441 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf438, buf441, 1024, 128, grid=grid(1024), stream=stream0)
        buf442 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf441, buf442, 256, 4, grid=grid(256), stream=stream0)
        buf443 = reinterpret_tensor(buf439, (1, 512, 1024), (524288, 1024, 1), 0); del buf439  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf443, addmm_11, 524288, grid=grid(524288), stream=stream0)
        del addmm_11
        buf444 = reinterpret_tensor(buf438, (512, 256), (256, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (512, 1024), (1024, 1), 0), permute_472, out=buf444)
        del permute_472
        buf445 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (1024, 512), (1, 1024), 0), view_42, out=buf445)
        del view_42
        buf446 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf443, buf446, 4096, 128, grid=grid(4096), stream=stream0)
        buf447 = reinterpret_tensor(buf441, (1, 1024), (1024, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf446, buf447, 1024, 4, grid=grid(1024), stream=stream0)
        buf450 = reinterpret_tensor(buf428, (1, 512, 256), (131072, 256, 1), 0); del buf428  # reuse
        buf453 = reinterpret_tensor(buf424, (1, 512, 256), (131072, 256, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf435, buf444, primals_32, mul_10, div_61, getitem_17, buf450, buf453, 512, 256, grid=grid(512), stream=stream0)
        del div_61
        del getitem_17
        del primals_32
        buf451 = empty((256, ), device='cuda', dtype=torch.float32)
        buf452 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf435, buf444, mul_10, buf451, buf452, 256, 512, grid=grid(256), stream=stream0)
        del buf435
        del mul_10
        buf454 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (512, 256), (256, 1), 0), permute_476, out=buf454)
        del permute_476
        buf455 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (256, 512), (1, 256), 0), view_40, out=buf455)
        del view_40
        buf456 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf453, buf456, 1024, 128, grid=grid(1024), stream=stream0)
        buf457 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf456, buf457, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf458 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf454, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_197, getitem_198, getitem_199, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_21
        del clone_default_30
        del clone_default_31
        del clone_default_32
        del getitem_197
        del getitem_198
        del getitem_199
        buf459 = buf458[0]
        buf460 = buf458[1]
        buf461 = buf458[2]
        del buf458
        buf462 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (512, 256), (256, 1), 0), permute_488, out=buf462)
        del permute_488
        buf463 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (256, 512), (1, 256), 0), view_24, out=buf463)
        buf464 = buf456; del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf461, buf464, 1024, 128, grid=grid(1024), stream=stream0)
        buf465 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf464, buf465, 256, 4, grid=grid(256), stream=stream0)
        buf466 = reinterpret_tensor(buf461, (512, 256), (256, 1), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (512, 256), (256, 1), 0), permute_493, out=buf466)
        del permute_493
        buf467 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (256, 512), (1, 256), 0), view_24, out=buf467)
        buf468 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf460, buf468, 1024, 128, grid=grid(1024), stream=stream0)
        buf469 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf468, buf469, 256, 4, grid=grid(256), stream=stream0)
        buf470 = reinterpret_tensor(buf460, (512, 256), (256, 1), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (512, 256), (256, 1), 0), permute_497, out=buf470)
        del permute_497
        buf471 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (256, 512), (1, 256), 0), view_24, out=buf471)
        del view_24
        buf472 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf459, buf472, 1024, 128, grid=grid(1024), stream=stream0)
        buf473 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf472, buf473, 256, 4, grid=grid(256), stream=stream0)
        buf477 = reinterpret_tensor(buf459, (1, 512, 256), (131072, 256, 1), 0); del buf459  # reuse
        buf480 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf450, buf462, buf466, buf470, primals_22, mul_8, div_63, getitem_11, buf477, buf480, 512, 256, grid=grid(512), stream=stream0)
        del div_63
        del getitem_11
        del primals_22
        buf478 = empty((256, ), device='cuda', dtype=torch.float32)
        buf479 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf450, buf462, buf466, buf470, mul_8, buf478, buf479, 256, 512, grid=grid(256), stream=stream0)
        del buf450
        del buf462
        del mul_8
        buf481 = reinterpret_tensor(buf443, (512, 1024), (1024, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (512, 256), (256, 1), 0), permute_501, out=buf481)
        del permute_501
        buf482 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (256, 512), (1, 256), 0), view_22, out=buf482)
        del view_22
        buf483 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf480, buf483, 1024, 128, grid=grid(1024), stream=stream0)
        buf484 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf483, buf484, 256, 4, grid=grid(256), stream=stream0)
        buf485 = reinterpret_tensor(buf481, (1, 512, 1024), (524288, 1024, 1), 0); del buf481  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf485, addmm_5, 524288, grid=grid(524288), stream=stream0)
        del addmm_5
        buf486 = reinterpret_tensor(buf480, (512, 256), (256, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (512, 1024), (1024, 1), 0), permute_505, out=buf486)
        del permute_505
        buf487 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (1024, 512), (1, 1024), 0), view_20, out=buf487)
        del view_20
        buf488 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf485, buf488, 4096, 128, grid=grid(4096), stream=stream0)
        del buf485
        buf489 = reinterpret_tensor(buf483, (1, 1024), (1024, 1), 0); del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf488, buf489, 1024, 4, grid=grid(1024), stream=stream0)
        del buf488
        buf492 = reinterpret_tensor(buf470, (1, 512, 256), (131072, 256, 1), 0); del buf470  # reuse
        buf495 = reinterpret_tensor(buf466, (1, 512, 256), (131072, 256, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf477, buf486, primals_16, mul_3, div_64, getitem_7, buf492, buf495, 512, 256, grid=grid(512), stream=stream0)
        del div_64
        del getitem_7
        del primals_16
        buf493 = empty((256, ), device='cuda', dtype=torch.float32)
        buf494 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf477, buf486, mul_3, buf493, buf494, 256, 512, grid=grid(256), stream=stream0)
        del buf477
        del mul_3
        buf496 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (512, 256), (256, 1), 0), permute_509, out=buf496)
        del permute_509
        buf497 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (256, 512), (1, 256), 0), view_18, out=buf497)
        del view_18
        buf498 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf495, buf498, 1024, 128, grid=grid(1024), stream=stream0)
        del buf495
        buf499 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf498, buf499, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf500 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf496, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_204, getitem_205, getitem_206, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_23
        del clone_default_33
        del clone_default_34
        del clone_default_35
        del getitem_204
        del getitem_205
        del getitem_206
        buf501 = buf500[0]
        buf502 = buf500[1]
        buf503 = buf500[2]
        del buf500
        buf504 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf503, (512, 256), (256, 1), 0), permute_521, out=buf504)
        del permute_521
        buf505 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf503, (256, 512), (1, 256), 0), view_2, out=buf505)
        buf506 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf503, buf506, 1024, 128, grid=grid(1024), stream=stream0)
        buf507 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf506, buf507, 256, 4, grid=grid(256), stream=stream0)
        buf508 = reinterpret_tensor(buf503, (512, 256), (256, 1), 0); del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf502, (512, 256), (256, 1), 0), permute_526, out=buf508)
        del permute_526
        buf509 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf502, (256, 512), (1, 256), 0), view_2, out=buf509)
        buf510 = buf506; del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf502, buf510, 1024, 128, grid=grid(1024), stream=stream0)
        buf511 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf510, buf511, 256, 4, grid=grid(256), stream=stream0)
        buf512 = reinterpret_tensor(buf502, (512, 256), (256, 1), 0); del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (512, 256), (256, 1), 0), permute_530, out=buf512)
        del permute_530
        buf513 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (256, 512), (1, 256), 0), view_2, out=buf513)
        del view_2
        buf514 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf501, buf514, 1024, 128, grid=grid(1024), stream=stream0)
        del buf501
        buf515 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf514, buf515, 256, 4, grid=grid(256), stream=stream0)
        buf516 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_16.run(buf516, buf504, buf508, buf512, 131072, grid=grid(131072), stream=stream0)
        del buf504
        del buf508
        del buf512
        buf517 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf516, (512, 256), (256, 1), 0), permute_534, out=buf517)
        del permute_534
        buf518 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf516, (256, 512), (1, 256), 0), view, out=buf518)
        del view
        buf519 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf516, buf519, 1024, 128, grid=grid(1024), stream=stream0)
        del buf516
        buf520 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf519, buf520, 256, 4, grid=grid(256), stream=stream0)
        del buf519
        buf527 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf531 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf535 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_loss], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_17.run(buf517, getitem_3, primals_4, mul_1, div_66, slice_4, expand, primals_204, buf527, buf531, buf535, 512, 128, grid=grid(512), stream=stream0)
        del div_66
        del primals_4
        buf524 = empty((128, ), device='cuda', dtype=torch.float32)
        buf525 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_18.run(buf517, getitem_3, mul_1, buf524, buf525, 128, 512, grid=grid(128), stream=stream0)
        del getitem_3
        del mul_1
        buf526 = buf517; del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_19.run(buf526, 65536, grid=grid(65536), stream=stream0)
        aten.index_put_(buf526, [slice_4], buf527, True)
        del buf527
        del slice_4
        buf530 = empty((2, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_20.run(buf530, 256, grid=grid(256), stream=stream0)
        aten.index_put_(buf530, [expand], buf531, True)
        del buf531
        del expand
        buf534 = empty((30522, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_21.run(buf534, 3906816, grid=grid(3906816), stream=stream0)
        aten.index_put_(buf534, [primals_204], buf535, True)
        del buf535
        del primals_204
        return (buf534, buf530, buf526, buf524, buf525, reinterpret_tensor(buf518, (256, 128), (128, 1), 0), reinterpret_tensor(buf520, (256, ), (1, ), 0), reinterpret_tensor(buf513, (256, 256), (256, 1), 0), reinterpret_tensor(buf515, (256, ), (1, ), 0), reinterpret_tensor(buf509, (256, 256), (256, 1), 0), reinterpret_tensor(buf511, (256, ), (1, ), 0), reinterpret_tensor(buf505, (256, 256), (256, 1), 0), reinterpret_tensor(buf507, (256, ), (1, ), 0), reinterpret_tensor(buf497, (256, 256), (256, 1), 0), reinterpret_tensor(buf499, (256, ), (1, ), 0), buf493, buf494, reinterpret_tensor(buf487, (1024, 256), (256, 1), 0), reinterpret_tensor(buf489, (1024, ), (1, ), 0), reinterpret_tensor(buf482, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf484, (256, ), (1, ), 0), buf478, buf479, reinterpret_tensor(buf471, (256, 256), (256, 1), 0), reinterpret_tensor(buf473, (256, ), (1, ), 0), reinterpret_tensor(buf467, (256, 256), (256, 1), 0), reinterpret_tensor(buf469, (256, ), (1, ), 0), reinterpret_tensor(buf463, (256, 256), (256, 1), 0), reinterpret_tensor(buf465, (256, ), (1, ), 0), reinterpret_tensor(buf455, (256, 256), (256, 1), 0), reinterpret_tensor(buf457, (256, ), (1, ), 0), buf451, buf452, reinterpret_tensor(buf445, (1024, 256), (256, 1), 0), reinterpret_tensor(buf447, (1024, ), (1, ), 0), reinterpret_tensor(buf440, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf442, (256, ), (1, ), 0), buf436, buf437, reinterpret_tensor(buf429, (256, 256), (256, 1), 0), reinterpret_tensor(buf431, (256, ), (1, ), 0), reinterpret_tensor(buf425, (256, 256), (256, 1), 0), reinterpret_tensor(buf427, (256, ), (1, ), 0), reinterpret_tensor(buf421, (256, 256), (256, 1), 0), reinterpret_tensor(buf423, (256, ), (1, ), 0), reinterpret_tensor(buf413, (256, 256), (256, 1), 0), reinterpret_tensor(buf415, (256, ), (1, ), 0), buf409, buf410, reinterpret_tensor(buf403, (1024, 256), (256, 1), 0), reinterpret_tensor(buf405, (1024, ), (1, ), 0), reinterpret_tensor(buf398, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf400, (256, ), (1, ), 0), buf394, buf395, reinterpret_tensor(buf387, (256, 256), (256, 1), 0), reinterpret_tensor(buf389, (256, ), (1, ), 0), reinterpret_tensor(buf383, (256, 256), (256, 1), 0), reinterpret_tensor(buf385, (256, ), (1, ), 0), reinterpret_tensor(buf379, (256, 256), (256, 1), 0), reinterpret_tensor(buf381, (256, ), (1, ), 0), reinterpret_tensor(buf371, (256, 256), (256, 1), 0), reinterpret_tensor(buf373, (256, ), (1, ), 0), buf367, buf368, reinterpret_tensor(buf361, (1024, 256), (256, 1), 0), reinterpret_tensor(buf363, (1024, ), (1, ), 0), reinterpret_tensor(buf356, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf358, (256, ), (1, ), 0), buf352, buf353, reinterpret_tensor(buf345, (256, 256), (256, 1), 0), reinterpret_tensor(buf347, (256, ), (1, ), 0), reinterpret_tensor(buf341, (256, 256), (256, 1), 0), reinterpret_tensor(buf343, (256, ), (1, ), 0), reinterpret_tensor(buf337, (256, 256), (256, 1), 0), reinterpret_tensor(buf339, (256, ), (1, ), 0), reinterpret_tensor(buf329, (256, 256), (256, 1), 0), reinterpret_tensor(buf331, (256, ), (1, ), 0), buf325, buf326, reinterpret_tensor(buf319, (1024, 256), (256, 1), 0), reinterpret_tensor(buf321, (1024, ), (1, ), 0), reinterpret_tensor(buf314, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf316, (256, ), (1, ), 0), buf310, buf311, reinterpret_tensor(buf303, (256, 256), (256, 1), 0), reinterpret_tensor(buf305, (256, ), (1, ), 0), reinterpret_tensor(buf299, (256, 256), (256, 1), 0), reinterpret_tensor(buf301, (256, ), (1, ), 0), reinterpret_tensor(buf295, (256, 256), (256, 1), 0), reinterpret_tensor(buf297, (256, ), (1, ), 0), reinterpret_tensor(buf287, (256, 256), (256, 1), 0), reinterpret_tensor(buf289, (256, ), (1, ), 0), buf283, buf284, reinterpret_tensor(buf277, (1024, 256), (256, 1), 0), reinterpret_tensor(buf279, (1024, ), (1, ), 0), reinterpret_tensor(buf272, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf274, (256, ), (1, ), 0), buf268, buf269, reinterpret_tensor(buf261, (256, 256), (256, 1), 0), reinterpret_tensor(buf263, (256, ), (1, ), 0), reinterpret_tensor(buf257, (256, 256), (256, 1), 0), reinterpret_tensor(buf259, (256, ), (1, ), 0), reinterpret_tensor(buf253, (256, 256), (256, 1), 0), reinterpret_tensor(buf255, (256, ), (1, ), 0), reinterpret_tensor(buf245, (256, 256), (256, 1), 0), reinterpret_tensor(buf247, (256, ), (1, ), 0), buf241, buf242, reinterpret_tensor(buf235, (1024, 256), (256, 1), 0), reinterpret_tensor(buf237, (1024, ), (1, ), 0), reinterpret_tensor(buf230, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf232, (256, ), (1, ), 0), buf226, buf227, reinterpret_tensor(buf219, (256, 256), (256, 1), 0), reinterpret_tensor(buf221, (256, ), (1, ), 0), reinterpret_tensor(buf215, (256, 256), (256, 1), 0), reinterpret_tensor(buf217, (256, ), (1, ), 0), reinterpret_tensor(buf211, (256, 256), (256, 1), 0), reinterpret_tensor(buf213, (256, ), (1, ), 0), reinterpret_tensor(buf203, (256, 256), (256, 1), 0), reinterpret_tensor(buf205, (256, ), (1, ), 0), buf199, buf200, reinterpret_tensor(buf193, (1024, 256), (256, 1), 0), reinterpret_tensor(buf195, (1024, ), (1, ), 0), reinterpret_tensor(buf188, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf190, (256, ), (1, ), 0), buf184, buf185, reinterpret_tensor(buf177, (256, 256), (256, 1), 0), reinterpret_tensor(buf179, (256, ), (1, ), 0), reinterpret_tensor(buf173, (256, 256), (256, 1), 0), reinterpret_tensor(buf175, (256, ), (1, ), 0), reinterpret_tensor(buf169, (256, 256), (256, 1), 0), reinterpret_tensor(buf171, (256, ), (1, ), 0), reinterpret_tensor(buf161, (256, 256), (256, 1), 0), reinterpret_tensor(buf163, (256, ), (1, ), 0), buf157, buf158, reinterpret_tensor(buf151, (1024, 256), (256, 1), 0), reinterpret_tensor(buf153, (1024, ), (1, ), 0), reinterpret_tensor(buf146, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf148, (256, ), (1, ), 0), buf142, buf143, reinterpret_tensor(buf135, (256, 256), (256, 1), 0), reinterpret_tensor(buf137, (256, ), (1, ), 0), reinterpret_tensor(buf131, (256, 256), (256, 1), 0), reinterpret_tensor(buf133, (256, ), (1, ), 0), reinterpret_tensor(buf127, (256, 256), (256, 1), 0), reinterpret_tensor(buf129, (256, ), (1, ), 0), reinterpret_tensor(buf119, (256, 256), (256, 1), 0), reinterpret_tensor(buf121, (256, ), (1, ), 0), buf115, buf116, reinterpret_tensor(buf109, (1024, 256), (256, 1), 0), reinterpret_tensor(buf111, (1024, ), (1, ), 0), reinterpret_tensor(buf104, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf106, (256, ), (1, ), 0), buf100, buf101, reinterpret_tensor(buf93, (256, 256), (256, 1), 0), reinterpret_tensor(buf95, (256, ), (1, ), 0), reinterpret_tensor(buf89, (256, 256), (256, 1), 0), reinterpret_tensor(buf91, (256, ), (1, ), 0), reinterpret_tensor(buf85, (256, 256), (256, 1), 0), reinterpret_tensor(buf87, (256, ), (1, ), 0), reinterpret_tensor(buf77, (256, 256), (256, 1), 0), reinterpret_tensor(buf79, (256, ), (1, ), 0), buf73, buf74, reinterpret_tensor(buf67, (1024, 256), (256, 1), 0), reinterpret_tensor(buf69, (1024, ), (1, ), 0), reinterpret_tensor(buf62, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf64, (256, ), (1, ), 0), buf58, buf59, reinterpret_tensor(buf51, (256, 256), (256, 1), 0), reinterpret_tensor(buf53, (256, ), (1, ), 0), reinterpret_tensor(buf47, (256, 256), (256, 1), 0), reinterpret_tensor(buf49, (256, ), (1, ), 0), reinterpret_tensor(buf43, (256, 256), (256, 1), 0), reinterpret_tensor(buf45, (256, ), (1, ), 0), reinterpret_tensor(buf35, (256, 256), (256, 1), 0), reinterpret_tensor(buf37, (256, ), (1, ), 0), buf31, buf32, reinterpret_tensor(buf25, (1024, 256), (256, 1), 0), reinterpret_tensor(buf27, (1024, ), (1, ), 0), reinterpret_tensor(buf20, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf22, (256, ), (1, ), 0), buf16, buf17, reinterpret_tensor(buf10, (2, 256), (256, 1), 0), reinterpret_tensor(buf12, (2, ), (1, ), 0), None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    expand = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    slice_4 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    mul_1 = rand_strided((1, 512, 128), (65536, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 512, 128), (65536, 128, 1), device='cuda:0', dtype=torch.bool)
    view = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_33 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_34 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_35 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_204 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_205 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_23 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_3 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_24 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_30 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_31 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_32 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_197 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_198 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_199 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_21 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_10 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_11 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_15 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_46 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_27 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_28 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_29 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_190 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_192 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_19 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_17 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_22 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_24 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_25 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_26 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_183 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_184 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_17 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_29 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_21 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_22 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_23 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_176 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_177 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_178 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_15 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_31 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_29 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_36 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_112 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_18 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_19 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_20 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_169 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_13 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_38 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_35 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_43 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_15 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_16 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_17 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_162 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_163 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_11 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_45 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_41 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_50 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_156 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_12 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_13 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_14 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_155 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_156 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_157 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_9 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_52 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_47 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_57 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_178 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_9 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_10 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_11 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_148 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_149 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_150 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_7 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_59 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_53 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_64 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_200 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_6 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_7 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_8 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_141 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_142 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_143 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_5 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_66 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_59 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_71 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_222 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default_3 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_4 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_5 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_134 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_136 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_3 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_73 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_65 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_78 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_244 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clone_default = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_1 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_2 = rand_strided((1, 4, 512, 64), (131072, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((1, 4, 512), (2048, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_128 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_1 = rand_strided((1, 4, 512, 64), (131072, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_80 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_71 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_264 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.bool)
    mul_85 = rand_strided((1, 512, 256), (131072, 256, 1), device='cuda:0', dtype=torch.float32)
    view_266 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    sub_39 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    sub_41 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_4 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_6 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_134 = rand_strided((2, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_158 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_196 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_229 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_262 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_328 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_332 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_336 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_356 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_361 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_394 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_422 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_427 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_468 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_472 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_488 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_493 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_497 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_501 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_505 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_526 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, primals_112, primals_118, primals_128, primals_134, primals_144, primals_150, primals_160, primals_166, primals_176, primals_182, primals_192, primals_198, primals_204, expand, slice_4, mul_1, getitem_3, view, view_2, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_18, getitem_7, mul_3, view_20, addmm_5, view_22, getitem_11, mul_8, view_24, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_40, getitem_17, mul_10, view_42, addmm_11, view_44, getitem_21, mul_15, view_46, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_62, getitem_27, mul_17, view_64, addmm_17, view_66, getitem_31, mul_22, view_68, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_84, getitem_37, mul_24, view_86, addmm_23, view_88, getitem_41, mul_29, view_90, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_106, getitem_47, mul_31, view_108, addmm_29, view_110, getitem_51, mul_36, view_112, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_128, getitem_57, mul_38, view_130, addmm_35, view_132, getitem_61, mul_43, view_134, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_150, getitem_67, mul_45, view_152, addmm_41, view_154, getitem_71, mul_50, view_156, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_172, getitem_77, mul_52, view_174, addmm_47, view_176, getitem_81, mul_57, view_178, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_194, getitem_87, mul_59, view_196, addmm_53, view_198, getitem_91, mul_64, view_200, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_216, getitem_97, mul_66, view_218, addmm_59, view_220, getitem_101, mul_71, view_222, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_238, getitem_107, mul_73, view_240, addmm_65, view_242, getitem_111, mul_78, view_244, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_260, getitem_117, mul_80, view_262, addmm_71, view_264, getitem_121, mul_85, view_266, sub_39, ne, sub_41, ne_3, ne_6, where_4, ne_8, where_6, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_158, permute_163, permute_167, div_33, permute_171, permute_175, div_34, permute_179, permute_191, permute_196, permute_200, div_36, permute_204, permute_208, div_37, permute_212, permute_224, permute_229, permute_233, div_39, permute_237, permute_241, div_40, permute_245, permute_257, permute_262, permute_266, div_42, permute_270, permute_274, div_43, permute_278, permute_290, permute_295, permute_299, div_45, permute_303, permute_307, div_46, permute_311, permute_323, permute_328, permute_332, div_48, permute_336, permute_340, div_49, permute_344, permute_356, permute_361, permute_365, div_51, permute_369, permute_373, div_52, permute_377, permute_389, permute_394, permute_398, div_54, permute_402, permute_406, div_55, permute_410, permute_422, permute_427, permute_431, div_57, permute_435, permute_439, div_58, permute_443, permute_455, permute_460, permute_464, div_60, permute_468, permute_472, div_61, permute_476, permute_488, permute_493, permute_497, div_63, permute_501, permute_505, div_64, permute_509, permute_521, permute_526, permute_530, permute_534, div_66, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ElectraForQuestionAnswering', benchmark_compiled_module)
