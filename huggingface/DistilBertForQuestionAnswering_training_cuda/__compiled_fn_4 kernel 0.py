
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


# kernel path: /tmp/torchinductor_youkaichao/dk/cdkjpkjfsdkexry6akyqkpmhtxzvgsrroiruskglyzdsybb4nf27.py
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
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/he/chexegi3ar4vmlflsv5nwobhiuoakcdgljahjgwbggmgs3qcodwz.py
# Source Nodes: [end_loss, start_loss], Original ATen: [aten._log_softmax_backward_data, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
# end_loss => convert_element_type_1, sum_11
# start_loss => convert_element_type, full_default_7, sum_8
triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*i1', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (0)).to(tl.int1)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp7 = tl.load(in_ptr3 + (0)).to(tl.int1)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp19 = tl.load(in_ptr4 + (r0), rmask, other=0.0)
    tmp20 = tl.load(in_ptr5 + (0)).to(tl.int1)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp22 = tl.load(in_ptr6 + (0)).to(tl.int1)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp5 = 2.0
    tmp6 = tmp4 / tmp5
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp6 / tmp10
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp24 = tmp23.to(tl.int64)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp6 / tmp25
    tmp27 = tl.where(tmp21, tmp26, tmp12)
    tmp28 = tmp19 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp18, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/ceh7svic445fhh6pjrfrcovoirb4fcp66m7upsjy3htje7cudsnl.py
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
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*i1', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4y63rwdk3s7clebiobdfk34ze3hp2ebwp5jvbgsnqa6dkbpcjz.py
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
    size_hints=[2, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (2*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5e/c5eibioybct3jfrf6ronaezxwamoqhxi6uw3njg54vpufjpr4bq7.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp7 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = 768.0
    tmp20 = tmp7 * tmp19
    tmp21 = tmp20 - tmp11
    tmp22 = tmp12 * tmp17
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp3
    tmp28 = tmp24 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp24, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp28, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpyve67lbyjin6btfoexgjhfekyxnayqqxqys66f5bz3n2dbevl.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/35/c35fnuguh6s4k2tx5oyv2rfwhpqjcemcvdhbnmsqwpf35j7ieokt.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clvhtiyzkj6slxirzgl2t5j6k2m54k26mesv7kr7p4wid3yjjpxp.py
# Source Nodes: [x_21], Original ATen: [aten.gelu, aten.gelu_backward]
# x_21 => add_41, erf_5, mul_40
triton_poi_fused_gelu_gelu_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: /tmp/torchinductor_youkaichao/d5/cd554sf7n4dhu5udqumlppjn4cme5ar5no6upb5ijs7rjtzxkuk3.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ay/cayvwhf4ixnegh2liekydcxazj3je453vd2ytst37avmgllbvz7m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp9 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 768.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjx75titkwaefdbrrrwmeedcs34tg4jo5zpz44k5mjhskxugxml.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kfvy2zrobgffbfgqbqo4rsmefyttb544hw26tybydprzhmgb2l.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/42/c4272cf7q35tsgjnk25mfpe47kku7zr6olgtq3yggf52dwt3gioo.py
# Source Nodes: [start_loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
# start_loss => full_default_7
triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
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
    tmp6 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp6 * tmp11
    tmp14 = tmp7 - tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp12, tmp15, tmp14)
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp16, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5fcwrfubknb34nhafiyq5ltiwkj3l2gptdbaqpzfz3byc4nplp.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hq/chqyxhgijh23wi3yksdzbjl7lf736dzxbxalkt3rserwyzxnvw7x.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]

triton_red_fused_add_native_layer_norm_backward_sum_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ue/cueegmhobo4ijcut63kix7skacoom6wgw2gsqzi7yhy6aoay7qn6.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqvmeno2egf3audgo7r4jmmttf4zgrdtodwftaiekf5mbytwlbb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
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
    tmp20 = 768.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.1111111111111112
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cah7la5eq3tv4xol6tvdjgptvmjb73pymrjnc637mqzrny2xibpy.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/df/cdfjhge75upz6izjkq6u7srlisx4bywg6e6xkpxuem6nsk2cbihf.py
# Source Nodes: [start_loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# start_loss => full_default_7
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 * tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp13 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 768.0
    tmp26 = tmp13 * tmp25
    tmp27 = tmp26 - tmp17
    tmp28 = tmp18 * tmp23
    tmp29 = tmp27 - tmp28
    tmp30 = tmp24 * tmp29
    tmp32 = tl.full([1], -1, tl.int64)
    tmp33 = tmp31 == tmp32
    tmp34 = 0.0
    tmp35 = tl.where(tmp33, tmp34, tmp30)
    tmp37 = tl.full([1], 0, tl.int64)
    tmp38 = tmp36 == tmp37
    tmp39 = tl.where(tmp38, tmp34, tmp30)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp11, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxo726u5n63juhe2hzvu7zeusez5kbmb5s5adbesofdah7t7beo.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dcyslck2q2obfd4wa5t42qkbqqrik4jwdvtv76xmrnrdzcku3k.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_20', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qc/cqc3ggcdlfljpww5mmfqjgvtu2j53xotlmvtck66kizhafycrcgr.py
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
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23440896
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
    primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_104, slice_2, mul, getitem_3, view, view_12, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_7, view_23, getitem_13, view_40, mul_9, view_42, addmm_10, view_44, getitem_17, mul_14, view_46, getitem_21, view_63, mul_16, view_65, addmm_16, view_67, getitem_25, mul_21, view_69, getitem_29, view_86, mul_23, view_88, addmm_22, view_90, getitem_33, mul_28, view_92, getitem_37, view_109, mul_30, view_111, addmm_28, view_113, getitem_41, mul_35, view_115, getitem_45, view_132, mul_37, view_134, addmm_34, view_136, getitem_49, mul_42, getitem_53, view_138, sub_20, ne, sub_22, ne_3, ne_6, where_10, ne_8, where_12, permute_67, div_18, permute_71, permute_75, div_19, permute_79, permute_84, permute_85, alias_10, permute_86, permute_87, permute_90, permute_95, permute_100, div_21, permute_104, permute_108, div_22, permute_112, permute_117, permute_118, alias_11, permute_119, permute_120, permute_123, permute_128, permute_133, div_24, permute_137, permute_141, div_25, permute_145, permute_150, permute_151, alias_12, permute_152, permute_153, permute_156, permute_161, permute_166, div_27, permute_170, permute_174, div_28, permute_178, permute_183, permute_184, alias_13, permute_185, permute_186, permute_189, permute_194, permute_199, div_30, permute_203, permute_207, div_31, permute_211, permute_216, permute_217, alias_14, permute_218, permute_219, permute_222, permute_227, permute_232, div_33, permute_236, permute_240, div_34, permute_244, permute_249, permute_250, alias_15, permute_251, permute_252, permute_255, permute_260, permute_265, div_36, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_104, (1, 128), (128, 1))
    assert_size_stride(slice_2, (1, 128), (512, 1))
    assert_size_stride(mul, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(getitem_3, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view, (128, 768), (768, 1))
    assert_size_stride(view_12, (1, 1, 1, 128), (128, 128, 128, 1))
    assert_size_stride(getitem_5, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_17, (128, 768), (768, 1))
    assert_size_stride(mul_2, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_19, (128, 768), (768, 1))
    assert_size_stride(addmm_4, (128, 3072), (3072, 1))
    assert_size_stride(view_21, (128, 3072), (3072, 1))
    assert_size_stride(getitem_9, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_7, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_23, (128, 768), (768, 1))
    assert_size_stride(getitem_13, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_40, (128, 768), (768, 1))
    assert_size_stride(mul_9, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_42, (128, 768), (768, 1))
    assert_size_stride(addmm_10, (128, 3072), (3072, 1))
    assert_size_stride(view_44, (128, 3072), (3072, 1))
    assert_size_stride(getitem_17, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_14, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_46, (128, 768), (768, 1))
    assert_size_stride(getitem_21, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_63, (128, 768), (768, 1))
    assert_size_stride(mul_16, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_65, (128, 768), (768, 1))
    assert_size_stride(addmm_16, (128, 3072), (3072, 1))
    assert_size_stride(view_67, (128, 3072), (3072, 1))
    assert_size_stride(getitem_25, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_21, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_69, (128, 768), (768, 1))
    assert_size_stride(getitem_29, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_86, (128, 768), (768, 1))
    assert_size_stride(mul_23, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_88, (128, 768), (768, 1))
    assert_size_stride(addmm_22, (128, 3072), (3072, 1))
    assert_size_stride(view_90, (128, 3072), (3072, 1))
    assert_size_stride(getitem_33, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_28, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_92, (128, 768), (768, 1))
    assert_size_stride(getitem_37, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_109, (128, 768), (768, 1))
    assert_size_stride(mul_30, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_111, (128, 768), (768, 1))
    assert_size_stride(addmm_28, (128, 3072), (3072, 1))
    assert_size_stride(view_113, (128, 3072), (3072, 1))
    assert_size_stride(getitem_41, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_35, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_115, (128, 768), (768, 1))
    assert_size_stride(getitem_45, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_132, (128, 768), (768, 1))
    assert_size_stride(mul_37, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_134, (128, 768), (768, 1))
    assert_size_stride(addmm_34, (128, 3072), (3072, 1))
    assert_size_stride(view_136, (128, 3072), (3072, 1))
    assert_size_stride(getitem_49, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_42, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(getitem_53, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_138, (128, 768), (768, 1))
    assert_size_stride(sub_20, (1, 128), (128, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_22, (1, 128), (128, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_10, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_12, (1, 1), (1, 1))
    assert_size_stride(permute_67, (2, 768), (768, 1))
    assert_size_stride(div_18, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_71, (768, 3072), (3072, 1))
    assert_size_stride(permute_75, (3072, 768), (768, 1))
    assert_size_stride(div_19, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_79, (768, 768), (768, 1))
    assert_size_stride(permute_84, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_85, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_10, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_86, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_87, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_90, (768, 768), (768, 1))
    assert_size_stride(permute_95, (768, 768), (768, 1))
    assert_size_stride(permute_100, (768, 768), (768, 1))
    assert_size_stride(div_21, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_104, (768, 3072), (3072, 1))
    assert_size_stride(permute_108, (3072, 768), (768, 1))
    assert_size_stride(div_22, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_112, (768, 768), (768, 1))
    assert_size_stride(permute_117, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_118, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_11, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_119, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_120, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_123, (768, 768), (768, 1))
    assert_size_stride(permute_128, (768, 768), (768, 1))
    assert_size_stride(permute_133, (768, 768), (768, 1))
    assert_size_stride(div_24, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_137, (768, 3072), (3072, 1))
    assert_size_stride(permute_141, (3072, 768), (768, 1))
    assert_size_stride(div_25, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_145, (768, 768), (768, 1))
    assert_size_stride(permute_150, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_151, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_12, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_152, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_153, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_156, (768, 768), (768, 1))
    assert_size_stride(permute_161, (768, 768), (768, 1))
    assert_size_stride(permute_166, (768, 768), (768, 1))
    assert_size_stride(div_27, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_170, (768, 3072), (3072, 1))
    assert_size_stride(permute_174, (3072, 768), (768, 1))
    assert_size_stride(div_28, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_178, (768, 768), (768, 1))
    assert_size_stride(permute_183, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_184, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_13, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_185, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_186, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_189, (768, 768), (768, 1))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(permute_199, (768, 768), (768, 1))
    assert_size_stride(div_30, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_203, (768, 3072), (3072, 1))
    assert_size_stride(permute_207, (3072, 768), (768, 1))
    assert_size_stride(div_31, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_211, (768, 768), (768, 1))
    assert_size_stride(permute_216, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_217, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_14, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_218, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_219, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_222, (768, 768), (768, 1))
    assert_size_stride(permute_227, (768, 768), (768, 1))
    assert_size_stride(permute_232, (768, 768), (768, 1))
    assert_size_stride(div_33, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_236, (768, 3072), (3072, 1))
    assert_size_stride(permute_240, (3072, 768), (768, 1))
    assert_size_stride(div_34, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_244, (768, 768), (768, 1))
    assert_size_stride(permute_249, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_250, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_15, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_251, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_252, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_255, (768, 768), (768, 1))
    assert_size_stride(permute_260, (768, 768), (768, 1))
    assert_size_stride(permute_265, (768, 768), (768, 1))
    assert_size_stride(div_36, (1, 128, 1), (128, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128), (128, 1))
    assert_size_stride(tangents_3, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_0.run(buf0, 128, grid=grid(128), stream=stream0)
        aten.scatter_(buf0,1,where_10,-1.0)
        del where_10
        buf4 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_0.run(buf4, 128, grid=grid(128), stream=stream0)
        aten.scatter_(buf4,1,where_12,-1.0)
        del where_12
        buf3 = empty((1, 1), device='cuda', dtype=torch.float32)
        buf7 = empty((1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_loss, start_loss], Original ATen: [aten._log_softmax_backward_data, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_1.run(buf0, ne_6, tangents_1, ne_3, buf4, ne_8, ne, buf3, buf7, 1, 128, grid=grid(1), stream=stream0)
        buf8 = empty((1, 128, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_2.run(tangents_2, buf4, ne_8, tangents_1, ne, sub_20, buf7, tangents_3, buf0, ne_6, ne_3, sub_22, buf3, buf8, 256, grid=grid(256), stream=stream0)
        del buf0
        del buf3
        del buf4
        del buf7
        del ne
        del ne_3
        del ne_6
        del ne_8
        del sub_20
        del sub_22
        del tangents_1
        del tangents_2
        del tangents_3
        buf9 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (128, 2), (2, 1), 0), permute_67, out=buf9)
        del permute_67
        buf10 = empty((2, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (2, 128), (1, 2), 0), view_138, out=buf10)
        del view_138
        buf11 = empty((1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf8, buf11, 2, 128, grid=grid(2), stream=stream0)
        del buf8
        buf14 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        buf17 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_4.run(buf9, getitem_53, primals_99, mul_42, div_18, getitem_49, buf14, buf17, 128, 768, grid=grid(128), stream=stream0)
        del div_18
        del getitem_49
        del primals_99
        buf15 = empty((768, ), device='cuda', dtype=torch.float32)
        buf16 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_5.run(buf9, getitem_53, mul_42, buf15, buf16, 768, 128, grid=grid(768), stream=stream0)
        del getitem_53
        del mul_42
        buf18 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (128, 768), (768, 1), 0), permute_71, out=buf18)
        del permute_71
        buf19 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (768, 128), (1, 768), 0), view_136, out=buf19)
        del view_136
        buf20 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf17, buf20, 768, 128, grid=grid(768), stream=stream0)
        buf21 = reinterpret_tensor(buf18, (1, 128, 3072), (393216, 3072, 1), 0); del buf18  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf21, addmm_34, 393216, grid=grid(393216), stream=stream0)
        del addmm_34
        buf22 = reinterpret_tensor(buf17, (128, 768), (768, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (128, 3072), (3072, 1), 0), permute_75, out=buf22)
        del permute_75
        buf23 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (3072, 128), (1, 3072), 0), view_134, out=buf23)
        del view_134
        buf24 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf21, buf24, 3072, 128, grid=grid(3072), stream=stream0)
        buf27 = reinterpret_tensor(buf9, (1, 128, 768), (98304, 768, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf14, buf22, primals_93, mul_37, div_19, buf27, 128, 768, grid=grid(128), stream=stream0)
        del div_19
        del primals_93
        buf28 = empty((768, ), device='cuda', dtype=torch.float32)
        buf29 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf14, buf22, mul_37, buf28, buf29, 768, 128, grid=grid(768), stream=stream0)
        del mul_37
        buf30 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (128, 768), (768, 1), 0), permute_79, out=buf30)
        del permute_79
        buf31 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (768, 128), (1, 768), 0), view_132, out=buf31)
        del view_132
        buf33 = reinterpret_tensor(buf14, (12, 128, 64), (8192, 64, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_84, reinterpret_tensor(buf30, (12, 128, 64), (64, 768, 1), 0), out=buf33)
        del permute_84
        buf39 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf33, buf39, 98304, grid=grid(98304), stream=stream0)
        buf40 = reinterpret_tensor(buf33, (128, 768), (768, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf39, permute_90, out=buf40)
        del permute_90
        buf34 = empty((12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf30, (12, 128, 64), (64, 768, 1), 0), permute_85, out=buf34)
        del permute_85
        buf36 = empty((1, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_12.run(buf34, getitem_45, alias_10, view_12, buf36, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_10
        del getitem_45
        buf37 = reinterpret_tensor(buf30, (12, 64, 128), (8192, 128, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_86, reinterpret_tensor(buf36, (12, 128, 128), (16384, 128, 1), 0), out=buf37)
        del permute_86
        buf43 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (128, 768), (1, 128), 0), permute_95, out=buf43)
        del permute_95
        buf38 = empty((12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (12, 128, 128), (16384, 128, 1), 0), permute_87, out=buf38)
        del permute_87
        buf46 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf38, buf46, 98304, grid=grid(98304), stream=stream0)
        buf47 = reinterpret_tensor(buf38, (128, 768), (768, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, permute_100, out=buf47)
        del permute_100
        buf32 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf54 = empty((768, ), device='cuda', dtype=torch.float32)
        buf55 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf27, buf40, buf43, buf47, mul_35, buf32, buf54, buf55, 768, 128, grid=grid(768), stream=stream0)
        buf41 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (768, 128), (1, 768), 0), view_115, out=buf41)
        buf42 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf39, buf42, 768, 128, grid=grid(768), stream=stream0)
        buf44 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (768, 128), (128, 1), 0), view_115, out=buf44)
        buf45 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf37, buf45, 768, 128, grid=grid(768), stream=stream0)
        buf48 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (768, 128), (1, 768), 0), view_115, out=buf48)
        del view_115
        buf49 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf46, buf49, 768, 128, grid=grid(768), stream=stream0)
        buf50 = buf27; del buf27  # reuse
        buf53 = reinterpret_tensor(buf46, (1, 128, 768), (98304, 768, 1), 0); del buf46  # reuse
        buf56 = reinterpret_tensor(buf37, (1, 128, 768), (98304, 768, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_16.run(buf50, buf40, buf43, buf47, primals_83, mul_35, div_21, getitem_41, buf53, buf56, 128, 768, grid=grid(128), stream=stream0)
        del div_21
        del getitem_41
        del mul_35
        del primals_83
        buf57 = reinterpret_tensor(buf21, (128, 3072), (3072, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (128, 768), (768, 1), 0), permute_104, out=buf57)
        del permute_104
        buf58 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (768, 128), (1, 768), 0), view_113, out=buf58)
        del view_113
        buf59 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf56, buf59, 768, 128, grid=grid(768), stream=stream0)
        buf60 = reinterpret_tensor(buf57, (1, 128, 3072), (393216, 3072, 1), 0); del buf57  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf60, addmm_28, 393216, grid=grid(393216), stream=stream0)
        del addmm_28
        buf61 = reinterpret_tensor(buf56, (128, 768), (768, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (128, 3072), (3072, 1), 0), permute_108, out=buf61)
        del permute_108
        buf62 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (3072, 128), (1, 3072), 0), view_111, out=buf62)
        del view_111
        buf63 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf60, buf63, 3072, 128, grid=grid(3072), stream=stream0)
        buf66 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf53, buf61, primals_77, mul_30, div_22, buf66, 128, 768, grid=grid(128), stream=stream0)
        del div_22
        del primals_77
        buf67 = empty((768, ), device='cuda', dtype=torch.float32)
        buf68 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf53, buf61, mul_30, buf67, buf68, 768, 128, grid=grid(768), stream=stream0)
        del mul_30
        buf69 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (128, 768), (768, 1), 0), permute_112, out=buf69)
        del permute_112
        buf70 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (768, 128), (1, 768), 0), view_109, out=buf70)
        del view_109
        buf72 = reinterpret_tensor(buf53, (12, 128, 64), (8192, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_117, reinterpret_tensor(buf69, (12, 128, 64), (64, 768, 1), 0), out=buf72)
        del permute_117
        buf78 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf72, buf78, 98304, grid=grid(98304), stream=stream0)
        buf79 = reinterpret_tensor(buf72, (128, 768), (768, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf78, permute_123, out=buf79)
        del permute_123
        buf73 = reinterpret_tensor(buf36, (12, 128, 128), (16384, 128, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf69, (12, 128, 64), (64, 768, 1), 0), permute_118, out=buf73)
        del permute_118
        buf75 = reinterpret_tensor(buf34, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf34  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_12.run(buf73, getitem_37, alias_11, view_12, buf75, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_11
        del getitem_37
        buf76 = reinterpret_tensor(buf69, (12, 64, 128), (8192, 128, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_119, reinterpret_tensor(buf75, (12, 128, 128), (16384, 128, 1), 0), out=buf76)
        del permute_119
        buf82 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (128, 768), (1, 128), 0), permute_128, out=buf82)
        del permute_128
        buf77 = reinterpret_tensor(buf40, (12, 128, 64), (8192, 64, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf75, (12, 128, 128), (16384, 128, 1), 0), permute_120, out=buf77)
        del permute_120
        buf85 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf77, buf85, 98304, grid=grid(98304), stream=stream0)
        buf86 = reinterpret_tensor(buf77, (128, 768), (768, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf85, permute_133, out=buf86)
        del permute_133
        buf71 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf93 = empty((768, ), device='cuda', dtype=torch.float32)
        buf94 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf66, buf79, buf82, buf86, mul_28, buf71, buf93, buf94, 768, 128, grid=grid(768), stream=stream0)
        buf80 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (768, 128), (1, 768), 0), view_92, out=buf80)
        buf81 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf78, buf81, 768, 128, grid=grid(768), stream=stream0)
        buf83 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (768, 128), (128, 1), 0), view_92, out=buf83)
        buf84 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf76, buf84, 768, 128, grid=grid(768), stream=stream0)
        buf87 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (768, 128), (1, 768), 0), view_92, out=buf87)
        del view_92
        buf88 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf85, buf88, 768, 128, grid=grid(768), stream=stream0)
        buf89 = buf66; del buf66  # reuse
        buf92 = reinterpret_tensor(buf85, (1, 128, 768), (98304, 768, 1), 0); del buf85  # reuse
        buf95 = reinterpret_tensor(buf76, (1, 128, 768), (98304, 768, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_16.run(buf89, buf79, buf82, buf86, primals_67, mul_28, div_24, getitem_33, buf92, buf95, 128, 768, grid=grid(128), stream=stream0)
        del div_24
        del getitem_33
        del mul_28
        del primals_67
        buf96 = reinterpret_tensor(buf60, (128, 3072), (3072, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf95, (128, 768), (768, 1), 0), permute_137, out=buf96)
        del permute_137
        buf97 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf95, (768, 128), (1, 768), 0), view_90, out=buf97)
        del view_90
        buf98 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf95, buf98, 768, 128, grid=grid(768), stream=stream0)
        buf99 = reinterpret_tensor(buf96, (1, 128, 3072), (393216, 3072, 1), 0); del buf96  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf99, addmm_22, 393216, grid=grid(393216), stream=stream0)
        del addmm_22
        buf100 = reinterpret_tensor(buf95, (128, 768), (768, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (128, 3072), (3072, 1), 0), permute_141, out=buf100)
        del permute_141
        buf101 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (3072, 128), (1, 3072), 0), view_88, out=buf101)
        del view_88
        buf102 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf99, buf102, 3072, 128, grid=grid(3072), stream=stream0)
        buf105 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf92, buf100, primals_61, mul_23, div_25, buf105, 128, 768, grid=grid(128), stream=stream0)
        del div_25
        del primals_61
        buf106 = empty((768, ), device='cuda', dtype=torch.float32)
        buf107 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf92, buf100, mul_23, buf106, buf107, 768, 128, grid=grid(768), stream=stream0)
        del mul_23
        buf108 = reinterpret_tensor(buf92, (128, 768), (768, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (128, 768), (768, 1), 0), permute_145, out=buf108)
        del permute_145
        buf109 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (768, 128), (1, 768), 0), view_86, out=buf109)
        del view_86
        buf111 = reinterpret_tensor(buf100, (12, 128, 64), (8192, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_150, reinterpret_tensor(buf108, (12, 128, 64), (64, 768, 1), 0), out=buf111)
        del permute_150
        buf117 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf111, buf117, 98304, grid=grid(98304), stream=stream0)
        buf118 = reinterpret_tensor(buf111, (128, 768), (768, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf117, permute_156, out=buf118)
        del permute_156
        buf112 = reinterpret_tensor(buf75, (12, 128, 128), (16384, 128, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf108, (12, 128, 64), (64, 768, 1), 0), permute_151, out=buf112)
        del permute_151
        buf114 = reinterpret_tensor(buf73, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf73  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_12.run(buf112, getitem_29, alias_12, view_12, buf114, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_12
        del getitem_29
        buf115 = reinterpret_tensor(buf108, (12, 64, 128), (8192, 128, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_152, reinterpret_tensor(buf114, (12, 128, 128), (16384, 128, 1), 0), out=buf115)
        del permute_152
        buf121 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (128, 768), (1, 128), 0), permute_161, out=buf121)
        del permute_161
        buf116 = reinterpret_tensor(buf79, (12, 128, 64), (8192, 64, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf114, (12, 128, 128), (16384, 128, 1), 0), permute_153, out=buf116)
        del permute_153
        buf124 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf116, buf124, 98304, grid=grid(98304), stream=stream0)
        buf125 = reinterpret_tensor(buf116, (128, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf124, permute_166, out=buf125)
        del permute_166
        buf110 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf132 = empty((768, ), device='cuda', dtype=torch.float32)
        buf133 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf105, buf118, buf121, buf125, mul_21, buf110, buf132, buf133, 768, 128, grid=grid(768), stream=stream0)
        buf119 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (768, 128), (1, 768), 0), view_69, out=buf119)
        buf120 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf117, buf120, 768, 128, grid=grid(768), stream=stream0)
        buf122 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (768, 128), (128, 1), 0), view_69, out=buf122)
        buf123 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf115, buf123, 768, 128, grid=grid(768), stream=stream0)
        buf126 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (768, 128), (1, 768), 0), view_69, out=buf126)
        del view_69
        buf127 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf124, buf127, 768, 128, grid=grid(768), stream=stream0)
        buf128 = buf105; del buf105  # reuse
        buf131 = reinterpret_tensor(buf124, (1, 128, 768), (98304, 768, 1), 0); del buf124  # reuse
        buf134 = reinterpret_tensor(buf115, (1, 128, 768), (98304, 768, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_16.run(buf128, buf118, buf121, buf125, primals_51, mul_21, div_27, getitem_25, buf131, buf134, 128, 768, grid=grid(128), stream=stream0)
        del div_27
        del getitem_25
        del mul_21
        del primals_51
        buf135 = reinterpret_tensor(buf99, (128, 3072), (3072, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (128, 768), (768, 1), 0), permute_170, out=buf135)
        del permute_170
        buf136 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (768, 128), (1, 768), 0), view_67, out=buf136)
        del view_67
        buf137 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf134, buf137, 768, 128, grid=grid(768), stream=stream0)
        buf138 = reinterpret_tensor(buf135, (1, 128, 3072), (393216, 3072, 1), 0); del buf135  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf138, addmm_16, 393216, grid=grid(393216), stream=stream0)
        del addmm_16
        buf139 = reinterpret_tensor(buf134, (128, 768), (768, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (128, 3072), (3072, 1), 0), permute_174, out=buf139)
        del permute_174
        buf140 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (3072, 128), (1, 3072), 0), view_65, out=buf140)
        del view_65
        buf141 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf138, buf141, 3072, 128, grid=grid(3072), stream=stream0)
        buf144 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf131, buf139, primals_45, mul_16, div_28, buf144, 128, 768, grid=grid(128), stream=stream0)
        del div_28
        del primals_45
        buf145 = empty((768, ), device='cuda', dtype=torch.float32)
        buf146 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf131, buf139, mul_16, buf145, buf146, 768, 128, grid=grid(768), stream=stream0)
        del mul_16
        buf147 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (128, 768), (768, 1), 0), permute_178, out=buf147)
        del permute_178
        buf148 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (768, 128), (1, 768), 0), view_63, out=buf148)
        del view_63
        buf150 = reinterpret_tensor(buf131, (12, 128, 64), (8192, 64, 1), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_183, reinterpret_tensor(buf147, (12, 128, 64), (64, 768, 1), 0), out=buf150)
        del permute_183
        buf156 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf150, buf156, 98304, grid=grid(98304), stream=stream0)
        buf157 = reinterpret_tensor(buf150, (128, 768), (768, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf156, permute_189, out=buf157)
        del permute_189
        buf151 = reinterpret_tensor(buf114, (12, 128, 128), (16384, 128, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf147, (12, 128, 64), (64, 768, 1), 0), permute_184, out=buf151)
        del permute_184
        buf153 = reinterpret_tensor(buf112, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf112  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_12.run(buf151, getitem_21, alias_13, view_12, buf153, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_13
        del getitem_21
        buf154 = reinterpret_tensor(buf147, (12, 64, 128), (8192, 128, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_185, reinterpret_tensor(buf153, (12, 128, 128), (16384, 128, 1), 0), out=buf154)
        del permute_185
        buf160 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (128, 768), (1, 128), 0), permute_194, out=buf160)
        del permute_194
        buf155 = reinterpret_tensor(buf118, (12, 128, 64), (8192, 64, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (12, 128, 128), (16384, 128, 1), 0), permute_186, out=buf155)
        del permute_186
        buf163 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf155, buf163, 98304, grid=grid(98304), stream=stream0)
        buf164 = reinterpret_tensor(buf155, (128, 768), (768, 1), 0); del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf163, permute_199, out=buf164)
        del permute_199
        buf149 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf171 = empty((768, ), device='cuda', dtype=torch.float32)
        buf172 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf144, buf157, buf160, buf164, mul_14, buf149, buf171, buf172, 768, 128, grid=grid(768), stream=stream0)
        buf158 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (768, 128), (1, 768), 0), view_46, out=buf158)
        buf159 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf156, buf159, 768, 128, grid=grid(768), stream=stream0)
        buf161 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (768, 128), (128, 1), 0), view_46, out=buf161)
        buf162 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf154, buf162, 768, 128, grid=grid(768), stream=stream0)
        buf165 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (768, 128), (1, 768), 0), view_46, out=buf165)
        del view_46
        buf166 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf163, buf166, 768, 128, grid=grid(768), stream=stream0)
        buf167 = buf144; del buf144  # reuse
        buf170 = reinterpret_tensor(buf163, (1, 128, 768), (98304, 768, 1), 0); del buf163  # reuse
        buf173 = reinterpret_tensor(buf154, (1, 128, 768), (98304, 768, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_16.run(buf167, buf157, buf160, buf164, primals_35, mul_14, div_30, getitem_17, buf170, buf173, 128, 768, grid=grid(128), stream=stream0)
        del div_30
        del getitem_17
        del mul_14
        del primals_35
        buf174 = reinterpret_tensor(buf138, (128, 3072), (3072, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (128, 768), (768, 1), 0), permute_203, out=buf174)
        del permute_203
        buf175 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (768, 128), (1, 768), 0), view_44, out=buf175)
        del view_44
        buf176 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf173, buf176, 768, 128, grid=grid(768), stream=stream0)
        buf177 = reinterpret_tensor(buf174, (1, 128, 3072), (393216, 3072, 1), 0); del buf174  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf177, addmm_10, 393216, grid=grid(393216), stream=stream0)
        del addmm_10
        buf178 = reinterpret_tensor(buf173, (128, 768), (768, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (128, 3072), (3072, 1), 0), permute_207, out=buf178)
        del permute_207
        buf179 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (3072, 128), (1, 3072), 0), view_42, out=buf179)
        del view_42
        buf180 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf177, buf180, 3072, 128, grid=grid(3072), stream=stream0)
        buf183 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf170, buf178, primals_29, mul_9, div_31, buf183, 128, 768, grid=grid(128), stream=stream0)
        del div_31
        del primals_29
        buf184 = empty((768, ), device='cuda', dtype=torch.float32)
        buf185 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf170, buf178, mul_9, buf184, buf185, 768, 128, grid=grid(768), stream=stream0)
        del mul_9
        buf186 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (128, 768), (768, 1), 0), permute_211, out=buf186)
        del permute_211
        buf187 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (768, 128), (1, 768), 0), view_40, out=buf187)
        del view_40
        buf189 = reinterpret_tensor(buf170, (12, 128, 64), (8192, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_216, reinterpret_tensor(buf186, (12, 128, 64), (64, 768, 1), 0), out=buf189)
        del permute_216
        buf195 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf189, buf195, 98304, grid=grid(98304), stream=stream0)
        buf196 = reinterpret_tensor(buf189, (128, 768), (768, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf195, permute_222, out=buf196)
        del permute_222
        buf190 = reinterpret_tensor(buf153, (12, 128, 128), (16384, 128, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf186, (12, 128, 64), (64, 768, 1), 0), permute_217, out=buf190)
        del permute_217
        buf192 = reinterpret_tensor(buf151, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf151  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_12.run(buf190, getitem_13, alias_14, view_12, buf192, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_14
        del getitem_13
        buf193 = reinterpret_tensor(buf186, (12, 64, 128), (8192, 128, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_218, reinterpret_tensor(buf192, (12, 128, 128), (16384, 128, 1), 0), out=buf193)
        del permute_218
        buf199 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (128, 768), (1, 128), 0), permute_227, out=buf199)
        del permute_227
        buf194 = reinterpret_tensor(buf157, (12, 128, 64), (8192, 64, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf192, (12, 128, 128), (16384, 128, 1), 0), permute_219, out=buf194)
        del permute_219
        buf202 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf194, buf202, 98304, grid=grid(98304), stream=stream0)
        buf203 = reinterpret_tensor(buf194, (128, 768), (768, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf202, permute_232, out=buf203)
        del permute_232
        buf188 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf210 = empty((768, ), device='cuda', dtype=torch.float32)
        buf211 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_14.run(buf183, buf196, buf199, buf203, mul_7, buf188, buf210, buf211, 768, 128, grid=grid(768), stream=stream0)
        buf197 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (768, 128), (1, 768), 0), view_23, out=buf197)
        buf198 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf195, buf198, 768, 128, grid=grid(768), stream=stream0)
        del buf195
        buf200 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (768, 128), (128, 1), 0), view_23, out=buf200)
        buf201 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf193, buf201, 768, 128, grid=grid(768), stream=stream0)
        buf204 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (768, 128), (1, 768), 0), view_23, out=buf204)
        del view_23
        buf205 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf202, buf205, 768, 128, grid=grid(768), stream=stream0)
        buf206 = buf183; del buf183  # reuse
        buf209 = reinterpret_tensor(buf202, (1, 128, 768), (98304, 768, 1), 0); del buf202  # reuse
        buf212 = reinterpret_tensor(buf193, (1, 128, 768), (98304, 768, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_16.run(buf206, buf196, buf199, buf203, primals_19, mul_7, div_33, getitem_9, buf209, buf212, 128, 768, grid=grid(128), stream=stream0)
        del div_33
        del getitem_9
        del mul_7
        del primals_19
        buf213 = reinterpret_tensor(buf177, (128, 3072), (3072, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (128, 768), (768, 1), 0), permute_236, out=buf213)
        del permute_236
        buf214 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (768, 128), (1, 768), 0), view_21, out=buf214)
        del view_21
        buf215 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf212, buf215, 768, 128, grid=grid(768), stream=stream0)
        buf216 = reinterpret_tensor(buf213, (1, 128, 3072), (393216, 3072, 1), 0); del buf213  # reuse
        # Source Nodes: [x_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf216, addmm_4, 393216, grid=grid(393216), stream=stream0)
        del addmm_4
        buf217 = reinterpret_tensor(buf212, (128, 768), (768, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (128, 3072), (3072, 1), 0), permute_240, out=buf217)
        del permute_240
        buf218 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (3072, 128), (1, 3072), 0), view_19, out=buf218)
        del view_19
        buf219 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf216, buf219, 3072, 128, grid=grid(3072), stream=stream0)
        buf222 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf209, buf217, primals_13, mul_2, div_34, buf222, 128, 768, grid=grid(128), stream=stream0)
        del div_34
        del primals_13
        buf223 = empty((768, ), device='cuda', dtype=torch.float32)
        buf224 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf209, buf217, mul_2, buf223, buf224, 768, 128, grid=grid(768), stream=stream0)
        del mul_2
        buf225 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (128, 768), (768, 1), 0), permute_244, out=buf225)
        del permute_244
        buf226 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (768, 128), (1, 768), 0), view_17, out=buf226)
        del view_17
        buf227 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17.run(buf222, buf227, 768, 128, grid=grid(768), stream=stream0)
        buf228 = reinterpret_tensor(buf209, (12, 128, 64), (8192, 64, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_249, reinterpret_tensor(buf225, (12, 128, 64), (64, 768, 1), 0), out=buf228)
        del permute_249
        buf229 = reinterpret_tensor(buf192, (12, 128, 128), (16384, 128, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (12, 128, 64), (64, 768, 1), 0), permute_250, out=buf229)
        del permute_250
        buf231 = reinterpret_tensor(buf190, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf190  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_12.run(buf229, getitem_5, alias_15, view_12, buf231, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_15
        del buf229
        del getitem_5
        del view_12
        buf232 = reinterpret_tensor(buf225, (12, 64, 128), (8192, 128, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_251, reinterpret_tensor(buf231, (12, 128, 128), (16384, 128, 1), 0), out=buf232)
        del permute_251
        buf233 = reinterpret_tensor(buf203, (12, 128, 64), (8192, 64, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf231, (12, 128, 128), (16384, 128, 1), 0), permute_252, out=buf233)
        del buf231
        del permute_252
        buf234 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf228, buf234, 98304, grid=grid(98304), stream=stream0)
        buf235 = reinterpret_tensor(buf228, (128, 768), (768, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf234, permute_255, out=buf235)
        del permute_255
        buf236 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (768, 128), (1, 768), 0), view, out=buf236)
        buf237 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf234, buf237, 768, 128, grid=grid(768), stream=stream0)
        buf238 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (128, 768), (1, 128), 0), permute_260, out=buf238)
        del permute_260
        buf239 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (768, 128), (128, 1), 0), view, out=buf239)
        buf240 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf232, buf240, 768, 128, grid=grid(768), stream=stream0)
        buf241 = reinterpret_tensor(buf232, (128, 768), (768, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf233, buf241, 98304, grid=grid(98304), stream=stream0)
        buf242 = reinterpret_tensor(buf233, (128, 768), (768, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf241, permute_265, out=buf242)
        del permute_265
        buf243 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf241, (768, 128), (1, 768), 0), view, out=buf243)
        del view
        buf244 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf241, buf244, 768, 128, grid=grid(768), stream=stream0)
        buf245 = buf222; del buf222  # reuse
        buf252 = reinterpret_tensor(buf241, (1, 128, 768), (98304, 768, 1), 0); del buf241  # reuse
        buf256 = reinterpret_tensor(buf196, (1, 128, 768), (98304, 768, 1), 0); del buf196  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_18.run(buf245, buf235, buf238, buf242, getitem_3, primals_3, mul, div_36, slice_2, primals_104, buf252, buf256, 128, 768, grid=grid(128), stream=stream0)
        del buf235
        del buf238
        del buf242
        del div_36
        del getitem_3
        del primals_3
        buf249 = empty((768, ), device='cuda', dtype=torch.float32)
        buf250 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_19.run(buf245, mul, buf249, buf250, 768, 128, grid=grid(768), stream=stream0)
        del buf245
        del mul
        buf251 = reinterpret_tensor(buf216, (512, 768), (768, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_20.run(buf251, 393216, grid=grid(393216), stream=stream0)
        aten.index_put_(buf251, [slice_2], buf252, True)
        del buf252
        del slice_2
        buf255 = empty((30522, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_21.run(buf255, 23440896, grid=grid(23440896), stream=stream0)
        aten.index_put_(buf255, [primals_104], buf256, True)
        del buf256
        del primals_104
        return (buf255, buf251, buf249, buf250, reinterpret_tensor(buf243, (768, 768), (768, 1), 0), reinterpret_tensor(buf244, (768, ), (1, ), 0), reinterpret_tensor(buf239, (768, 768), (768, 1), 0), reinterpret_tensor(buf240, (768, ), (1, ), 0), reinterpret_tensor(buf236, (768, 768), (768, 1), 0), reinterpret_tensor(buf237, (768, ), (1, ), 0), reinterpret_tensor(buf226, (768, 768), (768, 1), 0), reinterpret_tensor(buf227, (768, ), (1, ), 0), buf223, buf224, reinterpret_tensor(buf218, (3072, 768), (768, 1), 0), reinterpret_tensor(buf219, (3072, ), (1, ), 0), reinterpret_tensor(buf214, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf215, (768, ), (1, ), 0), buf210, buf211, reinterpret_tensor(buf204, (768, 768), (768, 1), 0), reinterpret_tensor(buf205, (768, ), (1, ), 0), reinterpret_tensor(buf200, (768, 768), (768, 1), 0), reinterpret_tensor(buf201, (768, ), (1, ), 0), reinterpret_tensor(buf197, (768, 768), (768, 1), 0), reinterpret_tensor(buf198, (768, ), (1, ), 0), reinterpret_tensor(buf187, (768, 768), (768, 1), 0), reinterpret_tensor(buf188, (768, ), (1, ), 0), buf184, buf185, reinterpret_tensor(buf179, (3072, 768), (768, 1), 0), reinterpret_tensor(buf180, (3072, ), (1, ), 0), reinterpret_tensor(buf175, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf176, (768, ), (1, ), 0), buf171, buf172, reinterpret_tensor(buf165, (768, 768), (768, 1), 0), reinterpret_tensor(buf166, (768, ), (1, ), 0), reinterpret_tensor(buf161, (768, 768), (768, 1), 0), reinterpret_tensor(buf162, (768, ), (1, ), 0), reinterpret_tensor(buf158, (768, 768), (768, 1), 0), reinterpret_tensor(buf159, (768, ), (1, ), 0), reinterpret_tensor(buf148, (768, 768), (768, 1), 0), reinterpret_tensor(buf149, (768, ), (1, ), 0), buf145, buf146, reinterpret_tensor(buf140, (3072, 768), (768, 1), 0), reinterpret_tensor(buf141, (3072, ), (1, ), 0), reinterpret_tensor(buf136, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf137, (768, ), (1, ), 0), buf132, buf133, reinterpret_tensor(buf126, (768, 768), (768, 1), 0), reinterpret_tensor(buf127, (768, ), (1, ), 0), reinterpret_tensor(buf122, (768, 768), (768, 1), 0), reinterpret_tensor(buf123, (768, ), (1, ), 0), reinterpret_tensor(buf119, (768, 768), (768, 1), 0), reinterpret_tensor(buf120, (768, ), (1, ), 0), reinterpret_tensor(buf109, (768, 768), (768, 1), 0), reinterpret_tensor(buf110, (768, ), (1, ), 0), buf106, buf107, reinterpret_tensor(buf101, (3072, 768), (768, 1), 0), reinterpret_tensor(buf102, (3072, ), (1, ), 0), reinterpret_tensor(buf97, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf98, (768, ), (1, ), 0), buf93, buf94, reinterpret_tensor(buf87, (768, 768), (768, 1), 0), reinterpret_tensor(buf88, (768, ), (1, ), 0), reinterpret_tensor(buf83, (768, 768), (768, 1), 0), reinterpret_tensor(buf84, (768, ), (1, ), 0), reinterpret_tensor(buf80, (768, 768), (768, 1), 0), reinterpret_tensor(buf81, (768, ), (1, ), 0), reinterpret_tensor(buf70, (768, 768), (768, 1), 0), reinterpret_tensor(buf71, (768, ), (1, ), 0), buf67, buf68, reinterpret_tensor(buf62, (3072, 768), (768, 1), 0), reinterpret_tensor(buf63, (3072, ), (1, ), 0), reinterpret_tensor(buf58, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf59, (768, ), (1, ), 0), buf54, buf55, reinterpret_tensor(buf48, (768, 768), (768, 1), 0), reinterpret_tensor(buf49, (768, ), (1, ), 0), reinterpret_tensor(buf44, (768, 768), (768, 1), 0), reinterpret_tensor(buf45, (768, ), (1, ), 0), reinterpret_tensor(buf41, (768, 768), (768, 1), 0), reinterpret_tensor(buf42, (768, ), (1, ), 0), reinterpret_tensor(buf31, (768, 768), (768, 1), 0), reinterpret_tensor(buf32, (768, ), (1, ), 0), buf28, buf29, reinterpret_tensor(buf23, (3072, 768), (768, 1), 0), reinterpret_tensor(buf24, (3072, ), (1, ), 0), reinterpret_tensor(buf19, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf20, (768, ), (1, ), 0), buf15, buf16, reinterpret_tensor(buf10, (2, 768), (768, 1), 0), reinterpret_tensor(buf11, (2, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    slice_2 = rand_strided((1, 128), (512, 1), device='cuda:0', dtype=torch.int64)
    mul = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    view = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((1, 1, 1, 128), (128, 128, 128, 1), device='cuda:0', dtype=torch.bool)
    getitem_5 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_17 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_2 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_7 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_40 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_14 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_46 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_63 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_21 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_86 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_23 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_33 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_28 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_92 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_109 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_30 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_111 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_113 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_35 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_45 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_132 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_37 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_136 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_49 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_42 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    view_138 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sub_20 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    sub_22 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_10 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_12 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_67 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_71 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_75 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_79 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_84 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_85 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_10 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_86 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_87 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_90 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_95 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_100 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_104 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_108 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_112 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_117 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_118 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_11 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_119 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_120 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_128 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_133 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_137 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_141 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_145 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_12 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_152 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_153 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_161 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_174 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_178 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_185 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_186 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_207 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_211 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_217 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_232 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_236 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_260 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_104, slice_2, mul, getitem_3, view, view_12, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_7, view_23, getitem_13, view_40, mul_9, view_42, addmm_10, view_44, getitem_17, mul_14, view_46, getitem_21, view_63, mul_16, view_65, addmm_16, view_67, getitem_25, mul_21, view_69, getitem_29, view_86, mul_23, view_88, addmm_22, view_90, getitem_33, mul_28, view_92, getitem_37, view_109, mul_30, view_111, addmm_28, view_113, getitem_41, mul_35, view_115, getitem_45, view_132, mul_37, view_134, addmm_34, view_136, getitem_49, mul_42, getitem_53, view_138, sub_20, ne, sub_22, ne_3, ne_6, where_10, ne_8, where_12, permute_67, div_18, permute_71, permute_75, div_19, permute_79, permute_84, permute_85, alias_10, permute_86, permute_87, permute_90, permute_95, permute_100, div_21, permute_104, permute_108, div_22, permute_112, permute_117, permute_118, alias_11, permute_119, permute_120, permute_123, permute_128, permute_133, div_24, permute_137, permute_141, div_25, permute_145, permute_150, permute_151, alias_12, permute_152, permute_153, permute_156, permute_161, permute_166, div_27, permute_170, permute_174, div_28, permute_178, permute_183, permute_184, alias_13, permute_185, permute_186, permute_189, permute_194, permute_199, div_30, permute_203, permute_207, div_31, permute_211, permute_216, permute_217, alias_14, permute_218, permute_219, permute_222, permute_227, permute_232, div_33, permute_236, permute_240, div_34, permute_244, permute_249, permute_250, alias_15, permute_251, permute_252, permute_255, permute_260, permute_265, div_36, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistilBertForQuestionAnswering', benchmark_compiled_module)
