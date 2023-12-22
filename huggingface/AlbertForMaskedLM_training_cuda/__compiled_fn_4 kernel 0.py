
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


# kernel path: /tmp/torchinductor_youkaichao/ca/ccarl5ypfeazjbf7hfw2cigauvecmnfznaljn4fkuqgrc2kgfuhc.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_1
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ji/cjivva35nrjxummuzug4vgfwcnmnn5qo5bkjhdvca6ghm43la35h.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_1
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvdkjua3h36i6td7gxukgcvepcdd6avomokh3zpuybilgfj2ibj.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 30000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (30000*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([1, 1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp8 = tmp5 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp8, tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp19 = tl.load(in_ptr2 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr4 + (r1 + (30000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr0 + (r1 + (30000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r1 + (30000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.full([1, 1], -100, tl.int64)
        tmp18 = tmp1 != tmp17
        tmp23 = tmp20 / tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp18, tmp23, tmp24)
        tmp26 = tmp16 * tmp25
        tmp28 = tl.exp(tmp27)
        tmp29 = tmp28 * tmp13
        tmp30 = tmp26 - tmp29
        tmp31 = tmp15 + tmp30
        tl.store(out_ptr1 + (r1 + (30000*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgjh74xqq3dsv4teqgw3gh4jfxm5sp5z2wfg5msilk3idm5zktxv.py
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
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30000
    rnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + (30000*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l6/cl6scx2uskesbrkh7g255ypc7bj6lrnm5aojdjtp2ayixtq3clpi.py
# Source Nodes: [add_62, hidden_states_38, hidden_states_39, mul_49], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.pow, aten.tanh_backward]
# add_62 => add_113
# hidden_states_38 => mul_102
# hidden_states_39 => mul_103, sub_38
# mul_49 => mul_99
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp10 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 * tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tmp2 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = 128.0
    tmp24 = tmp16 / tmp23
    tmp25 = tmp2 * tmp23
    tmp26 = tmp25 - tmp6
    tmp27 = tmp17 * tmp22
    tmp28 = tmp26 - tmp27
    tmp29 = tmp24 * tmp28
    tmp30 = tmp29 * tmp9
    tmp31 = tmp10 * tmp10
    tmp32 = tmp11 - tmp31
    tmp33 = tmp30 * tmp32
    tmp34 = 0.7978845608028654
    tmp35 = tmp33 * tmp34
    tmp36 = 0.044715
    tmp37 = tmp35 * tmp36
    tmp38 = tmp7 * tmp7
    tmp39 = 3.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp37 * tmp40
    tmp42 = tmp35 + tmp41
    tmp43 = tmp29 * tmp12
    tmp44 = tmp43 * tmp8
    tmp45 = tmp42 + tmp44
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp45, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rj/crjgd5r4n7o5u2ujzxdsuefsglqrsvr7oadqw7sbla7rsm2d7gtx.py
# Source Nodes: [add_62, hidden_states_38, hidden_states_39, mul_49], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_62 => add_113
# hidden_states_38 => mul_102
# hidden_states_39 => mul_103, sub_38
# mul_49 => mul_99
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp4 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 * tmp6
    tmp9 = tmp7 - tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3t/c3tnbovky5svyoipiyubxoib62pslzx4neuil2uwpz7yhvekoqai.py
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
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4l2ba5emtxuppqqde6qcynasahemh4fouf4kilv5caemalyjpik.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2edrs52qsz3wsvett67la5tkf6ypnvb2puslqkfqeciyfczoc6.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/tf/ctfhb6fbii7lr4q5qsh44m2edt2b2v2q5g3eu244rastshpqtrmk.py
# Source Nodes: [add_59, mul_45], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
# add_59 => add_108
# mul_45 => mul_93
triton_poi_fused_add_mul_pow_tanh_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/tt/cttomqf5pytcpwhkso27nyhk3zw35mpoldohdkxnovfayogb5zpm.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_10', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/g6/cg6dtjgau36wngdrsxwhw52tf76pn7rnwxtkqunrem7poa4cbas2.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_11', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/o5/co5o2hg7yd4duzqv4bpmqddhqju23bq4fnxu7brdz4hdgdkzduh2.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]

triton_per_fused__softmax_backward_data_div_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_div_12', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/b3/cb34nszswvqbzc4gotktqzxubzfnctxogcugd3vhca3e2kvuowzp.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_13', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/km/ckmkxh5ktneljm5owkzaoljmtymtwfgiogi4j6jblgpyq2vzr5eu.py
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: 'i32', 60: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(59, 60))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
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


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2ospii62uhxzewp4dfa7kdjejoayhtpsi6b6nrt2vu2wfbk5eq.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_15', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/lq/clqgpcyvlodtcx3wugk6lzgrexm22vpu6d4j5mfgiaxbzeew7546.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/na/cnaaf56o4fhycrixdp3h2xzjnx7eceqoorwykme5uaf3hgppa233.py
# Source Nodes: [], Original ATen: [aten.add, aten.sum]

triton_per_fused_add_sum_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_sum_17', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/xs/cxsjghwvqqfsmdhbsue7jujcgmjdakaycaibqysxh4imqrlhadcb.py
# Source Nodes: [], Original ATen: [aten.add, aten.sum]

triton_red_fused_add_sum_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_sum_18', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5g4mqek7sey3cbznpufaru3plntgs4ccgtuq6yhunga4b3dvba.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
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


# kernel path: /tmp/torchinductor_youkaichao/l6/cl6mvdumhue523s6i74mmrbnrcn5feb7fvbafhnqiu4s4pw4exz2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_20', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/we/cwe6hrsqbfbgxayaty77oxgktygfpgmbphtdwcaowbredfoncnys.py
# Source Nodes: [], Original ATen: [aten.add, aten.sum]

triton_per_fused_add_sum_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_sum_21', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/pu/cpudynfwyexmtjmfocfewdwz4oteyjkwdtnbpo3p4yy643sq2boq.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_22', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/it/citvnnqakksro5c4t35kcyj334lpmjvwx6fnol2qhuipirnjz7sw.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_23', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/5i/c5i3snhrepzs6idlgrwa7n6u44qgsloji6ve5shg25bmwsh4lijt.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_24', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/4d/c4dvh5u4ky7qpqpp2kp3kgsuvmczwzwup3lpfjp4dj7f45u6phyo.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_25', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/6k/c6k5kebkzni26i6qyizb5trbphs6zg4hwiisiuub43m6lliwzsj5.py
# Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_per_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_26', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/nd/cndalhlq4qcuifpjnl2grbxkfa46ailt6koj2aml4tcbr5dfc7ph.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_27', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbyczepmwou27j7mli6yo5dlipkdf32ogs3hojfcybnaookmjgf.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_28', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xz/cxz3zgoamsk5vlyln2uxvra3d5ew332isdgxl47nv4ifqavwqr5x.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_29', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jg/cjglqjo4jsk3jp6jatu2sxrdgpxbpvpgghpjuepegyq5lweci7xy.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_30', 'mutated_arg_names': []},
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
    primals_4, primals_16, primals_22, primals_26, primals_32, primals_33, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, addmm_73, tanh_12, getitem_51, rsqrt_25, view_268, sub_40, convert_element_type, permute_135, permute_139, div_27, permute_143, permute_147, div_28, permute_151, permute_156, permute_157, alias_29, permute_158, permute_159, permute_164, permute_168, permute_172, div_30, div_31, permute_189, permute_190, alias_31, permute_191, permute_192, div_33, div_34, permute_222, permute_223, alias_33, permute_224, permute_225, div_36, div_37, permute_255, permute_256, alias_35, permute_257, permute_258, div_39, div_40, permute_288, permute_289, alias_37, permute_290, permute_291, div_42, div_43, permute_321, permute_322, alias_39, permute_323, permute_324, div_45, div_46, permute_354, permute_355, alias_41, permute_356, permute_357, div_48, div_49, permute_387, permute_388, alias_43, permute_389, permute_390, div_51, div_52, permute_420, permute_421, alias_45, permute_422, permute_423, div_54, div_55, permute_453, permute_454, alias_47, permute_455, permute_456, div_57, div_58, permute_486, permute_487, alias_49, permute_488, permute_489, div_60, div_61, permute_519, permute_520, alias_51, permute_521, permute_522, permute_539, div_63, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_16, (4096, ), (1, ))
    assert_size_stride(primals_22, (4096, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_32, (1, 512), (512, 1))
    assert_size_stride(primals_33, (1, 512), (512, 1))
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
    assert_size_stride(addmm_73, (512, 128), (128, 1))
    assert_size_stride(tanh_12, (1, 512, 128), (65536, 128, 1))
    assert_size_stride(getitem_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(rsqrt_25, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_268, (512, 128), (128, 1))
    assert_size_stride(sub_40, (512, 30000), (30000, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_135, (30000, 128), (128, 1))
    assert_size_stride(permute_139, (128, 4096), (4096, 1))
    assert_size_stride(div_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_143, (4096, 16384), (16384, 1))
    assert_size_stride(permute_147, (16384, 4096), (4096, 1))
    assert_size_stride(div_28, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_151, (4096, 4096), (4096, 1))
    assert_size_stride(permute_156, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_157, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_29, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_158, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_159, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(permute_164, (4096, 4096), (4096, 1))
    assert_size_stride(permute_168, (4096, 4096), (4096, 1))
    assert_size_stride(permute_172, (4096, 4096), (4096, 1))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_31, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_189, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_190, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_31, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_191, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_192, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_34, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_222, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_223, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_33, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_224, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_225, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_37, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_255, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_256, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_35, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_257, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_258, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_288, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_289, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_37, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_290, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_291, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_321, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_322, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_39, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_323, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_324, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_354, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_355, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_41, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_356, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_357, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_387, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_388, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_43, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_389, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_390, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_420, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_421, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_45, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_422, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_423, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_453, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_454, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_47, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_455, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_456, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_486, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_487, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_49, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_488, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_489, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_519, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_520, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_51, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_521, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_522, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(permute_539, (4096, 128), (128, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 30000), (15360000, 30000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((512, 30000), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 15360000, grid=grid(15360000), stream=stream0)
        buf1 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_33, buf1, 512, grid=grid(512), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf5 = empty((1, 512, 30000), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_33, tangents_1, convert_element_type, tangents_2, sub_40, buf5, 512, 30000, grid=grid(512), stream=stream0)
        del buf0
        del convert_element_type
        del primals_33
        del sub_40
        del tangents_1
        del tangents_2
        buf6 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 30000), (30000, 1), 0), permute_135, out=buf6)
        del permute_135
        buf7 = empty((30000, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (30000, 512), (1, 30000), 0), view_268, out=buf7)
        del view_268
        buf8 = empty((1, 30000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf5, buf8, 30000, 512, grid=grid(30000), stream=stream0)
        del buf5
        buf11 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf14 = buf11; del buf11  # reuse
        # Source Nodes: [add_62, hidden_states_38, hidden_states_39, mul_49], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.pow, aten.tanh_backward]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4.run(buf14, buf6, primals_26, addmm_73, tanh_12, getitem_51, rsqrt_25, 512, 128, grid=grid(512), stream=stream0)
        del primals_26
        buf12 = empty((128, ), device='cuda', dtype=torch.float32)
        buf13 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_62, hidden_states_38, hidden_states_39, mul_49], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_5.run(buf6, addmm_73, tanh_12, getitem_51, rsqrt_25, buf12, buf13, 128, 512, grid=grid(128), stream=stream0)
        del addmm_73
        del getitem_51
        del rsqrt_25
        del tanh_12
        buf15 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (512, 128), (128, 1), 0), permute_139, out=buf15)
        del permute_139
        buf16 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (128, 512), (1, 128), 0), view_266, out=buf16)
        del view_266
        buf17 = empty_strided((1, 128, 4), (512, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf14, buf17, 512, 128, grid=grid(512), stream=stream0)
        buf18 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf17, buf18, 128, 4, grid=grid(128), stream=stream0)
        del buf17
        buf21 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf15, primals_22, mul_97, div_27, buf21, 512, 4096, grid=grid(512), stream=stream0)
        del div_27
        buf24 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (512, 4096), (4096, 1), 0), permute_143, out=buf24)
        buf28 = reinterpret_tensor(buf24, (1, 512, 16384), (8388608, 16384, 1), 0); del buf24  # reuse
        # Source Nodes: [add_59, mul_45], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf28, addmm_71, tanh_11, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_71
        del tanh_11
        buf29 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 16384), (16384, 1), 0), permute_147, out=buf29)
        buf34 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf21, buf29, primals_16, mul_91, div_28, buf34, 512, 4096, grid=grid(512), stream=stream0)
        del div_28
        buf37 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (512, 4096), (4096, 1), 0), permute_151, out=buf37)
        buf41 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_156, reinterpret_tensor(buf37, (64, 512, 64), (64, 4096, 1), 0), out=buf41)
        del permute_156
        buf47 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf41, buf47, 2097152, grid=grid(2097152), stream=stream0)
        buf48 = reinterpret_tensor(buf41, (512, 4096), (4096, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf47, permute_164, out=buf48)
        buf42 = empty((64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (64, 512, 64), (64, 4096, 1), 0), permute_157, out=buf42)
        del permute_157
        buf44 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf42, alias_29, buf44, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_29
        buf45 = reinterpret_tensor(buf37, (64, 64, 512), (32768, 512, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_158, reinterpret_tensor(buf44, (64, 512, 512), (262144, 512, 1), 0), out=buf45)
        del permute_158
        buf52 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (512, 4096), (1, 512), 0), permute_168, out=buf52)
        buf46 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf44, (64, 512, 512), (262144, 512, 1), 0), permute_159, out=buf46)
        del permute_159
        buf55 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf46, buf55, 2097152, grid=grid(2097152), stream=stream0)
        buf56 = reinterpret_tensor(buf46, (512, 4096), (4096, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf55, permute_172, out=buf56)
        buf60 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf63 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf34, buf48, buf52, buf56, primals_22, mul_89, div_30, buf60, buf63, 512, 4096, grid=grid(512), stream=stream0)
        del div_30
        buf66 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (512, 4096), (4096, 1), 0), permute_143, out=buf66)
        buf70 = reinterpret_tensor(buf66, (1, 512, 16384), (8388608, 16384, 1), 0); del buf66  # reuse
        # Source Nodes: [add_54, mul_41], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf70, addmm_65, tanh_10, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_65
        del tanh_10
        buf71 = reinterpret_tensor(buf60, (512, 4096), (4096, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 16384), (16384, 1), 0), permute_147, out=buf71)
        buf76 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf63, buf71, primals_16, mul_83, div_31, buf76, 512, 4096, grid=grid(512), stream=stream0)
        del div_31
        buf79 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (512, 4096), (4096, 1), 0), permute_151, out=buf79)
        buf83 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_189, reinterpret_tensor(buf79, (64, 512, 64), (64, 4096, 1), 0), out=buf83)
        del permute_189
        buf89 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf83, buf89, 2097152, grid=grid(2097152), stream=stream0)
        buf90 = reinterpret_tensor(buf83, (512, 4096), (4096, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf89, permute_164, out=buf90)
        buf84 = reinterpret_tensor(buf44, (64, 512, 512), (262144, 512, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (64, 512, 64), (64, 4096, 1), 0), permute_190, out=buf84)
        del permute_190
        buf86 = reinterpret_tensor(buf42, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf84, alias_31, buf86, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_31
        buf87 = reinterpret_tensor(buf79, (64, 64, 512), (32768, 512, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_191, reinterpret_tensor(buf86, (64, 512, 512), (262144, 512, 1), 0), out=buf87)
        del permute_191
        buf94 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (512, 4096), (1, 512), 0), permute_168, out=buf94)
        buf88 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf86, (64, 512, 512), (262144, 512, 1), 0), permute_192, out=buf88)
        del permute_192
        buf97 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf88, buf97, 2097152, grid=grid(2097152), stream=stream0)
        buf98 = reinterpret_tensor(buf88, (512, 4096), (4096, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf97, permute_172, out=buf98)
        buf102 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf105 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf76, buf90, buf94, buf98, primals_22, mul_81, div_33, buf102, buf105, 512, 4096, grid=grid(512), stream=stream0)
        del div_33
        buf108 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (512, 4096), (4096, 1), 0), permute_143, out=buf108)
        buf112 = reinterpret_tensor(buf108, (1, 512, 16384), (8388608, 16384, 1), 0); del buf108  # reuse
        # Source Nodes: [add_49, mul_37], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf112, addmm_59, tanh_9, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_59
        del tanh_9
        buf113 = reinterpret_tensor(buf102, (512, 4096), (4096, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (512, 16384), (16384, 1), 0), permute_147, out=buf113)
        buf118 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf105, buf113, primals_16, mul_75, div_34, buf118, 512, 4096, grid=grid(512), stream=stream0)
        del div_34
        buf121 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf118, (512, 4096), (4096, 1), 0), permute_151, out=buf121)
        buf125 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_222, reinterpret_tensor(buf121, (64, 512, 64), (64, 4096, 1), 0), out=buf125)
        del permute_222
        buf131 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf125, buf131, 2097152, grid=grid(2097152), stream=stream0)
        buf132 = reinterpret_tensor(buf125, (512, 4096), (4096, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf131, permute_164, out=buf132)
        buf126 = reinterpret_tensor(buf86, (64, 512, 512), (262144, 512, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (64, 512, 64), (64, 4096, 1), 0), permute_223, out=buf126)
        del permute_223
        buf128 = reinterpret_tensor(buf84, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf126, alias_33, buf128, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_33
        buf129 = reinterpret_tensor(buf121, (64, 64, 512), (32768, 512, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_224, reinterpret_tensor(buf128, (64, 512, 512), (262144, 512, 1), 0), out=buf129)
        del permute_224
        buf136 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (512, 4096), (1, 512), 0), permute_168, out=buf136)
        buf130 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf128, (64, 512, 512), (262144, 512, 1), 0), permute_225, out=buf130)
        del permute_225
        buf139 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf130, buf139, 2097152, grid=grid(2097152), stream=stream0)
        buf140 = reinterpret_tensor(buf130, (512, 4096), (4096, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf139, permute_172, out=buf140)
        buf144 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf147 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf118, buf132, buf136, buf140, primals_22, mul_73, div_36, buf144, buf147, 512, 4096, grid=grid(512), stream=stream0)
        del div_36
        buf150 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (512, 4096), (4096, 1), 0), permute_143, out=buf150)
        buf154 = reinterpret_tensor(buf150, (1, 512, 16384), (8388608, 16384, 1), 0); del buf150  # reuse
        # Source Nodes: [add_44, mul_33], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf154, addmm_53, tanh_8, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_53
        del tanh_8
        buf155 = reinterpret_tensor(buf144, (512, 4096), (4096, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (512, 16384), (16384, 1), 0), permute_147, out=buf155)
        buf160 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf147, buf155, primals_16, mul_67, div_37, buf160, 512, 4096, grid=grid(512), stream=stream0)
        del div_37
        buf163 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (512, 4096), (4096, 1), 0), permute_151, out=buf163)
        buf167 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_255, reinterpret_tensor(buf163, (64, 512, 64), (64, 4096, 1), 0), out=buf167)
        del permute_255
        buf173 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf167, buf173, 2097152, grid=grid(2097152), stream=stream0)
        buf174 = reinterpret_tensor(buf167, (512, 4096), (4096, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf173, permute_164, out=buf174)
        buf168 = reinterpret_tensor(buf128, (64, 512, 512), (262144, 512, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf163, (64, 512, 64), (64, 4096, 1), 0), permute_256, out=buf168)
        del permute_256
        buf170 = reinterpret_tensor(buf126, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf168, alias_35, buf170, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_35
        buf171 = reinterpret_tensor(buf163, (64, 64, 512), (32768, 512, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_257, reinterpret_tensor(buf170, (64, 512, 512), (262144, 512, 1), 0), out=buf171)
        del permute_257
        buf178 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (512, 4096), (1, 512), 0), permute_168, out=buf178)
        buf172 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf170, (64, 512, 512), (262144, 512, 1), 0), permute_258, out=buf172)
        del permute_258
        buf181 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf172, buf181, 2097152, grid=grid(2097152), stream=stream0)
        buf182 = reinterpret_tensor(buf172, (512, 4096), (4096, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf181, permute_172, out=buf182)
        buf186 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf189 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf160, buf174, buf178, buf182, primals_22, mul_65, div_39, buf186, buf189, 512, 4096, grid=grid(512), stream=stream0)
        del div_39
        buf192 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (512, 4096), (4096, 1), 0), permute_143, out=buf192)
        buf196 = reinterpret_tensor(buf192, (1, 512, 16384), (8388608, 16384, 1), 0); del buf192  # reuse
        # Source Nodes: [add_39, mul_29], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf196, addmm_47, tanh_7, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_47
        del tanh_7
        buf197 = reinterpret_tensor(buf186, (512, 4096), (4096, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (512, 16384), (16384, 1), 0), permute_147, out=buf197)
        buf202 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf189, buf197, primals_16, mul_59, div_40, buf202, 512, 4096, grid=grid(512), stream=stream0)
        del div_40
        buf205 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (512, 4096), (4096, 1), 0), permute_151, out=buf205)
        buf209 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_288, reinterpret_tensor(buf205, (64, 512, 64), (64, 4096, 1), 0), out=buf209)
        del permute_288
        buf215 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf209, buf215, 2097152, grid=grid(2097152), stream=stream0)
        buf216 = reinterpret_tensor(buf209, (512, 4096), (4096, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf215, permute_164, out=buf216)
        buf210 = reinterpret_tensor(buf170, (64, 512, 512), (262144, 512, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf205, (64, 512, 64), (64, 4096, 1), 0), permute_289, out=buf210)
        del permute_289
        buf212 = reinterpret_tensor(buf168, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf210, alias_37, buf212, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_37
        buf213 = reinterpret_tensor(buf205, (64, 64, 512), (32768, 512, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_290, reinterpret_tensor(buf212, (64, 512, 512), (262144, 512, 1), 0), out=buf213)
        del permute_290
        buf220 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (512, 4096), (1, 512), 0), permute_168, out=buf220)
        buf214 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf212, (64, 512, 512), (262144, 512, 1), 0), permute_291, out=buf214)
        del permute_291
        buf223 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf214, buf223, 2097152, grid=grid(2097152), stream=stream0)
        buf224 = reinterpret_tensor(buf214, (512, 4096), (4096, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf223, permute_172, out=buf224)
        buf228 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf231 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf202, buf216, buf220, buf224, primals_22, mul_57, div_42, buf228, buf231, 512, 4096, grid=grid(512), stream=stream0)
        del div_42
        buf234 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (512, 4096), (4096, 1), 0), permute_143, out=buf234)
        buf238 = reinterpret_tensor(buf234, (1, 512, 16384), (8388608, 16384, 1), 0); del buf234  # reuse
        # Source Nodes: [add_34, mul_25], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf238, addmm_41, tanh_6, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_41
        del tanh_6
        buf239 = reinterpret_tensor(buf228, (512, 4096), (4096, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (512, 16384), (16384, 1), 0), permute_147, out=buf239)
        buf244 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf231, buf239, primals_16, mul_51, div_43, buf244, 512, 4096, grid=grid(512), stream=stream0)
        del div_43
        buf247 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (512, 4096), (4096, 1), 0), permute_151, out=buf247)
        buf251 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_321, reinterpret_tensor(buf247, (64, 512, 64), (64, 4096, 1), 0), out=buf251)
        del permute_321
        buf257 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf251, buf257, 2097152, grid=grid(2097152), stream=stream0)
        buf258 = reinterpret_tensor(buf251, (512, 4096), (4096, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf257, permute_164, out=buf258)
        buf252 = reinterpret_tensor(buf212, (64, 512, 512), (262144, 512, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf247, (64, 512, 64), (64, 4096, 1), 0), permute_322, out=buf252)
        del permute_322
        buf254 = reinterpret_tensor(buf210, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf252, alias_39, buf254, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_39
        buf255 = reinterpret_tensor(buf247, (64, 64, 512), (32768, 512, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_323, reinterpret_tensor(buf254, (64, 512, 512), (262144, 512, 1), 0), out=buf255)
        del permute_323
        buf262 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (512, 4096), (1, 512), 0), permute_168, out=buf262)
        buf256 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf254, (64, 512, 512), (262144, 512, 1), 0), permute_324, out=buf256)
        del permute_324
        buf265 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf256, buf265, 2097152, grid=grid(2097152), stream=stream0)
        buf266 = reinterpret_tensor(buf256, (512, 4096), (4096, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf265, permute_172, out=buf266)
        buf270 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf273 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf244, buf258, buf262, buf266, primals_22, mul_49, div_45, buf270, buf273, 512, 4096, grid=grid(512), stream=stream0)
        del div_45
        buf276 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (512, 4096), (4096, 1), 0), permute_143, out=buf276)
        buf280 = reinterpret_tensor(buf276, (1, 512, 16384), (8388608, 16384, 1), 0); del buf276  # reuse
        # Source Nodes: [add_29, mul_21], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf280, addmm_35, tanh_5, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_35
        del tanh_5
        buf281 = reinterpret_tensor(buf270, (512, 4096), (4096, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (512, 16384), (16384, 1), 0), permute_147, out=buf281)
        buf286 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf273, buf281, primals_16, mul_43, div_46, buf286, 512, 4096, grid=grid(512), stream=stream0)
        del div_46
        buf289 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (512, 4096), (4096, 1), 0), permute_151, out=buf289)
        buf293 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_354, reinterpret_tensor(buf289, (64, 512, 64), (64, 4096, 1), 0), out=buf293)
        del permute_354
        buf299 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf293, buf299, 2097152, grid=grid(2097152), stream=stream0)
        buf300 = reinterpret_tensor(buf293, (512, 4096), (4096, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf299, permute_164, out=buf300)
        buf294 = reinterpret_tensor(buf254, (64, 512, 512), (262144, 512, 1), 0); del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf289, (64, 512, 64), (64, 4096, 1), 0), permute_355, out=buf294)
        del permute_355
        buf296 = reinterpret_tensor(buf252, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf294, alias_41, buf296, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_41
        buf297 = reinterpret_tensor(buf289, (64, 64, 512), (32768, 512, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_356, reinterpret_tensor(buf296, (64, 512, 512), (262144, 512, 1), 0), out=buf297)
        del permute_356
        buf304 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (512, 4096), (1, 512), 0), permute_168, out=buf304)
        buf298 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (64, 512, 512), (262144, 512, 1), 0), permute_357, out=buf298)
        del permute_357
        buf307 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf298, buf307, 2097152, grid=grid(2097152), stream=stream0)
        buf308 = reinterpret_tensor(buf298, (512, 4096), (4096, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf307, permute_172, out=buf308)
        buf312 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf315 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf286, buf300, buf304, buf308, primals_22, mul_41, div_48, buf312, buf315, 512, 4096, grid=grid(512), stream=stream0)
        del div_48
        buf318 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (512, 4096), (4096, 1), 0), permute_143, out=buf318)
        buf322 = reinterpret_tensor(buf318, (1, 512, 16384), (8388608, 16384, 1), 0); del buf318  # reuse
        # Source Nodes: [add_24, mul_17], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf322, addmm_29, tanh_4, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_29
        del tanh_4
        buf323 = reinterpret_tensor(buf312, (512, 4096), (4096, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (512, 16384), (16384, 1), 0), permute_147, out=buf323)
        buf328 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf315, buf323, primals_16, mul_35, div_49, buf328, 512, 4096, grid=grid(512), stream=stream0)
        del div_49
        buf331 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf328, (512, 4096), (4096, 1), 0), permute_151, out=buf331)
        buf335 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_387, reinterpret_tensor(buf331, (64, 512, 64), (64, 4096, 1), 0), out=buf335)
        del permute_387
        buf341 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf335, buf341, 2097152, grid=grid(2097152), stream=stream0)
        buf342 = reinterpret_tensor(buf335, (512, 4096), (4096, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf341, permute_164, out=buf342)
        buf336 = reinterpret_tensor(buf296, (64, 512, 512), (262144, 512, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf331, (64, 512, 64), (64, 4096, 1), 0), permute_388, out=buf336)
        del permute_388
        buf338 = reinterpret_tensor(buf294, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf336, alias_43, buf338, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_43
        buf339 = reinterpret_tensor(buf331, (64, 64, 512), (32768, 512, 1), 0); del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_389, reinterpret_tensor(buf338, (64, 512, 512), (262144, 512, 1), 0), out=buf339)
        del permute_389
        buf346 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (512, 4096), (1, 512), 0), permute_168, out=buf346)
        buf340 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf338, (64, 512, 512), (262144, 512, 1), 0), permute_390, out=buf340)
        del permute_390
        buf349 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf340, buf349, 2097152, grid=grid(2097152), stream=stream0)
        buf350 = reinterpret_tensor(buf340, (512, 4096), (4096, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf349, permute_172, out=buf350)
        buf354 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf357 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf328, buf342, buf346, buf350, primals_22, mul_33, div_51, buf354, buf357, 512, 4096, grid=grid(512), stream=stream0)
        del div_51
        buf362 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (512, 4096), (4096, 1), 0), permute_143, out=buf362)
        buf368 = reinterpret_tensor(buf362, (1, 512, 16384), (8388608, 16384, 1), 0); del buf362  # reuse
        # Source Nodes: [add_19, mul_13], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf368, addmm_23, tanh_3, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_23
        del tanh_3
        buf369 = reinterpret_tensor(buf354, (512, 4096), (4096, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (512, 16384), (16384, 1), 0), permute_147, out=buf369)
        buf376 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf357, buf369, primals_16, mul_27, div_52, buf376, 512, 4096, grid=grid(512), stream=stream0)
        del div_52
        buf381 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (512, 4096), (4096, 1), 0), permute_151, out=buf381)
        buf387 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_420, reinterpret_tensor(buf381, (64, 512, 64), (64, 4096, 1), 0), out=buf387)
        del permute_420
        buf393 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf387, buf393, 2097152, grid=grid(2097152), stream=stream0)
        buf394 = reinterpret_tensor(buf387, (512, 4096), (4096, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf393, permute_164, out=buf394)
        buf388 = reinterpret_tensor(buf338, (64, 512, 512), (262144, 512, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf381, (64, 512, 64), (64, 4096, 1), 0), permute_421, out=buf388)
        del permute_421
        buf390 = reinterpret_tensor(buf336, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf388, alias_45, buf390, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_45
        buf391 = reinterpret_tensor(buf381, (64, 64, 512), (32768, 512, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_422, reinterpret_tensor(buf390, (64, 512, 512), (262144, 512, 1), 0), out=buf391)
        del permute_422
        buf400 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (512, 4096), (1, 512), 0), permute_168, out=buf400)
        buf392 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf390, (64, 512, 512), (262144, 512, 1), 0), permute_423, out=buf392)
        del permute_423
        buf405 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf392, buf405, 2097152, grid=grid(2097152), stream=stream0)
        buf406 = reinterpret_tensor(buf392, (512, 4096), (4096, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf405, permute_172, out=buf406)
        buf412 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf415 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf376, buf394, buf400, buf406, primals_22, mul_25, div_54, buf412, buf415, 512, 4096, grid=grid(512), stream=stream0)
        del div_54
        buf418 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (512, 4096), (4096, 1), 0), permute_143, out=buf418)
        buf422 = reinterpret_tensor(buf418, (1, 512, 16384), (8388608, 16384, 1), 0); del buf418  # reuse
        # Source Nodes: [add_14, mul_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf422, addmm_17, tanh_2, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_17
        del tanh_2
        buf423 = reinterpret_tensor(buf412, (512, 4096), (4096, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (512, 16384), (16384, 1), 0), permute_147, out=buf423)
        buf428 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf415, buf423, primals_16, mul_19, div_55, buf428, 512, 4096, grid=grid(512), stream=stream0)
        del div_55
        buf431 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (512, 4096), (4096, 1), 0), permute_151, out=buf431)
        buf435 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_453, reinterpret_tensor(buf431, (64, 512, 64), (64, 4096, 1), 0), out=buf435)
        del permute_453
        buf441 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf435, buf441, 2097152, grid=grid(2097152), stream=stream0)
        buf442 = reinterpret_tensor(buf435, (512, 4096), (4096, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf441, permute_164, out=buf442)
        buf436 = reinterpret_tensor(buf390, (64, 512, 512), (262144, 512, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf431, (64, 512, 64), (64, 4096, 1), 0), permute_454, out=buf436)
        del permute_454
        buf438 = reinterpret_tensor(buf388, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf436, alias_47, buf438, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_47
        buf439 = reinterpret_tensor(buf431, (64, 64, 512), (32768, 512, 1), 0); del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_455, reinterpret_tensor(buf438, (64, 512, 512), (262144, 512, 1), 0), out=buf439)
        del permute_455
        buf446 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (512, 4096), (1, 512), 0), permute_168, out=buf446)
        buf440 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf438, (64, 512, 512), (262144, 512, 1), 0), permute_456, out=buf440)
        del permute_456
        buf449 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf440, buf449, 2097152, grid=grid(2097152), stream=stream0)
        buf450 = reinterpret_tensor(buf440, (512, 4096), (4096, 1), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf449, permute_172, out=buf450)
        buf454 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        buf457 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_13.run(buf428, buf442, buf446, buf450, primals_22, mul_17, div_57, buf454, buf457, 512, 4096, grid=grid(512), stream=stream0)
        del div_57
        buf460 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (512, 4096), (4096, 1), 0), permute_143, out=buf460)
        buf464 = reinterpret_tensor(buf460, (1, 512, 16384), (8388608, 16384, 1), 0); del buf460  # reuse
        # Source Nodes: [add_9, mul_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf464, addmm_11, tanh_1, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_11
        del tanh_1
        buf465 = reinterpret_tensor(buf454, (512, 4096), (4096, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (512, 16384), (16384, 1), 0), permute_147, out=buf465)
        buf470 = empty((1, 512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf457, buf465, primals_16, mul_11, div_58, buf470, 512, 4096, grid=grid(512), stream=stream0)
        del div_58
        buf473 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf470, (512, 4096), (4096, 1), 0), permute_151, out=buf473)
        buf477 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_486, reinterpret_tensor(buf473, (64, 512, 64), (64, 4096, 1), 0), out=buf477)
        del permute_486
        buf483 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf477, buf483, 2097152, grid=grid(2097152), stream=stream0)
        buf484 = reinterpret_tensor(buf477, (512, 4096), (4096, 1), 0); del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf483, permute_164, out=buf484)
        buf478 = reinterpret_tensor(buf438, (64, 512, 512), (262144, 512, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf473, (64, 512, 64), (64, 4096, 1), 0), permute_487, out=buf478)
        del permute_487
        buf480 = reinterpret_tensor(buf436, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf478, alias_49, buf480, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_49
        buf481 = reinterpret_tensor(buf473, (64, 64, 512), (32768, 512, 1), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_488, reinterpret_tensor(buf480, (64, 512, 512), (262144, 512, 1), 0), out=buf481)
        del permute_488
        buf488 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf481, (512, 4096), (1, 512), 0), permute_168, out=buf488)
        buf482 = empty((64, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf480, (64, 512, 512), (262144, 512, 1), 0), permute_489, out=buf482)
        del permute_489
        buf491 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf482, buf491, 2097152, grid=grid(2097152), stream=stream0)
        buf492 = reinterpret_tensor(buf482, (512, 4096), (4096, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf491, permute_172, out=buf492)
        buf106 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf107 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf360 = buf106; del buf106  # reuse
        buf502 = buf360; del buf360  # reuse
        buf361 = buf107; del buf107  # reuse
        buf503 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_14.run(buf502, buf503, buf15, mul_97, buf34, buf48, buf52, buf56, mul_89, buf76, buf90, buf94, buf98, mul_81, buf118, buf132, buf136, buf140, mul_73, buf160, buf174, buf178, buf182, mul_65, buf202, buf216, buf220, buf224, mul_57, buf376, buf394, buf400, buf406, mul_25, buf244, buf258, buf262, buf266, mul_49, buf428, buf442, buf446, buf450, mul_17, buf286, buf300, buf304, buf308, mul_41, buf470, buf484, buf488, buf492, mul_9, buf328, buf342, buf346, buf350, mul_33, 4096, 512, grid=grid(4096), stream=stream0)
        del buf132
        del buf136
        del buf140
        del buf15
        del buf174
        del buf178
        del buf182
        del buf216
        del buf220
        del buf224
        del buf258
        del buf262
        del buf266
        del buf300
        del buf304
        del buf308
        del buf342
        del buf346
        del buf350
        del buf394
        del buf400
        del buf406
        del buf442
        del buf446
        del buf450
        del buf48
        del buf52
        del buf56
        del buf90
        del buf94
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
        buf25 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (4096, 512), (1, 4096), 0), view_264, out=buf25)
        del view_264
        buf26 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf21, buf26, 16384, 128, grid=grid(16384), stream=stream0)
        buf110 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf105, buf110, 16384, 128, grid=grid(16384), stream=stream0)
        buf152 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf147, buf152, 16384, 128, grid=grid(16384), stream=stream0)
        buf194 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf189, buf194, 16384, 128, grid=grid(16384), stream=stream0)
        buf236 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf231, buf236, 16384, 128, grid=grid(16384), stream=stream0)
        buf278 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf273, buf278, 16384, 128, grid=grid(16384), stream=stream0)
        buf320 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf315, buf320, 16384, 128, grid=grid(16384), stream=stream0)
        buf364 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf357, buf364, 16384, 128, grid=grid(16384), stream=stream0)
        buf420 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf415, buf420, 16384, 128, grid=grid(16384), stream=stream0)
        buf462 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf457, buf462, 16384, 128, grid=grid(16384), stream=stream0)
        buf496 = reinterpret_tensor(buf484, (1, 512, 4096), (2097152, 4096, 1), 0); del buf484  # reuse
        buf499 = reinterpret_tensor(buf98, (1, 512, 4096), (2097152, 4096, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_16.run(buf496, buf470, buf488, buf492, primals_22, mul_9, div_60, buf499, 512, 4096, grid=grid(512), stream=stream0)
        del buf488
        del buf492
        del div_60
        del mul_9
        del primals_22
        buf506 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf499, buf506, 16384, 128, grid=grid(16384), stream=stream0)
        buf68 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf63, buf68, 16384, 128, grid=grid(16384), stream=stream0)
        buf111 = empty((1, 4096), device='cuda', dtype=torch.float32)
        buf366 = reinterpret_tensor(buf111, (4096, ), (1, ), 0); del buf111  # reuse
        buf508 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_17.run(buf508, buf26, buf68, buf110, buf152, buf194, buf236, buf420, buf278, buf462, buf320, buf506, buf364, 4096, 4, grid=grid(4096), stream=stream0)
        buf30 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (16384, 512), (1, 16384), 0), view_262, out=buf30)
        del view_262
        buf504 = empty((512, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (512, 4096), (4096, 1), 0), permute_143, out=buf504)
        del permute_143
        buf510 = reinterpret_tensor(buf504, (1, 512, 16384), (8388608, 16384, 1), 0); del buf504  # reuse
        # Source Nodes: [add_4, mul_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_9.run(buf510, addmm_5, tanh, 8388608, grid=grid(8388608), stream=stream0)
        del addmm_5
        del tanh
        buf115 = reinterpret_tensor(buf68, (1, 16384), (16384, 1), 0); del buf68  # reuse
        buf372 = reinterpret_tensor(buf115, (16384, ), (1, ), 0); del buf115  # reuse
        buf514 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_red_fused_add_sum_18.run(buf514, buf28, buf70, buf112, buf154, buf196, buf238, buf422, buf280, buf464, buf322, buf510, buf368, 16384, 512, grid=grid(16384), stream=stream0)
        del buf28
        buf511 = reinterpret_tensor(buf496, (512, 4096), (4096, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf510, (512, 16384), (16384, 1), 0), permute_147, out=buf511)
        del permute_147
        buf119 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf120 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf379 = buf119; del buf119  # reuse
        buf521 = buf379; del buf379  # reuse
        buf380 = buf120; del buf120  # reuse
        buf522 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf521, buf522, buf21, buf29, mul_91, buf63, buf71, mul_83, buf105, buf113, mul_75, buf147, buf155, mul_67, buf189, buf197, mul_59, buf231, buf239, mul_51, buf415, buf423, mul_19, buf273, buf281, mul_43, buf457, buf465, mul_11, buf315, buf323, mul_35, buf499, buf511, mul_3, buf357, buf369, mul_27, 4096, 512, grid=grid(4096), stream=stream0)
        del buf113
        del buf155
        del buf197
        del buf21
        del buf239
        del buf281
        del buf29
        del buf323
        del buf369
        del buf423
        del buf465
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
        buf38 = reinterpret_tensor(buf480, (4096, 4096), (4096, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (4096, 512), (1, 4096), 0), view_260, out=buf38)
        del view_260
        buf39 = buf506; del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf34, buf39, 16384, 128, grid=grid(16384), stream=stream0)
        buf123 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf118, buf123, 16384, 128, grid=grid(16384), stream=stream0)
        buf165 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf160, buf165, 16384, 128, grid=grid(16384), stream=stream0)
        buf207 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf202, buf207, 16384, 128, grid=grid(16384), stream=stream0)
        buf249 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf244, buf249, 16384, 128, grid=grid(16384), stream=stream0)
        buf291 = buf278; del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf286, buf291, 16384, 128, grid=grid(16384), stream=stream0)
        buf333 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf328, buf333, 16384, 128, grid=grid(16384), stream=stream0)
        buf383 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf376, buf383, 16384, 128, grid=grid(16384), stream=stream0)
        buf433 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf428, buf433, 16384, 128, grid=grid(16384), stream=stream0)
        buf475 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf470, buf475, 16384, 128, grid=grid(16384), stream=stream0)
        buf518 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_10.run(buf499, buf511, primals_16, mul_3, div_61, buf518, 512, 4096, grid=grid(512), stream=stream0)
        del div_61
        del mul_3
        del primals_16
        buf525 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf518, buf525, 16384, 128, grid=grid(16384), stream=stream0)
        buf81 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf76, buf81, 16384, 128, grid=grid(16384), stream=stream0)
        buf124 = empty((1, 4096), device='cuda', dtype=torch.float32)
        buf385 = reinterpret_tensor(buf124, (4096, ), (1, ), 0); del buf124  # reuse
        buf527 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_17.run(buf527, buf39, buf81, buf123, buf165, buf207, buf249, buf433, buf291, buf475, buf333, buf525, buf383, 4096, 4, grid=grid(4096), stream=stream0)
        buf49 = reinterpret_tensor(buf478, (4096, 4096), (4096, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (4096, 512), (1, 4096), 0), view_244, out=buf49)
        buf50 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf47, buf50, 16384, 128, grid=grid(16384), stream=stream0)
        buf134 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf131, buf134, 16384, 128, grid=grid(16384), stream=stream0)
        buf176 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf173, buf176, 16384, 128, grid=grid(16384), stream=stream0)
        buf218 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf215, buf218, 16384, 128, grid=grid(16384), stream=stream0)
        buf260 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf257, buf260, 16384, 128, grid=grid(16384), stream=stream0)
        buf302 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf299, buf302, 16384, 128, grid=grid(16384), stream=stream0)
        buf344 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf341, buf344, 16384, 128, grid=grid(16384), stream=stream0)
        buf396 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf393, buf396, 16384, 128, grid=grid(16384), stream=stream0)
        buf444 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf441, buf444, 16384, 128, grid=grid(16384), stream=stream0)
        buf486 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf483, buf486, 16384, 128, grid=grid(16384), stream=stream0)
        buf523 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (512, 4096), (4096, 1), 0), permute_151, out=buf523)
        del permute_151
        buf529 = reinterpret_tensor(buf511, (64, 512, 64), (32768, 64, 1), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_519, reinterpret_tensor(buf523, (64, 512, 64), (64, 4096, 1), 0), out=buf529)
        del permute_519
        buf535 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf529, buf535, 2097152, grid=grid(2097152), stream=stream0)
        del buf529
        buf538 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf535, buf538, 16384, 128, grid=grid(16384), stream=stream0)
        buf92 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf89, buf92, 16384, 128, grid=grid(16384), stream=stream0)
        buf135 = empty((1, 4096), device='cuda', dtype=torch.float32)
        buf398 = reinterpret_tensor(buf135, (4096, ), (1, ), 0); del buf135  # reuse
        buf540 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_17.run(buf540, buf50, buf92, buf134, buf176, buf218, buf260, buf444, buf302, buf486, buf344, buf538, buf396, 4096, 4, grid=grid(4096), stream=stream0)
        buf53 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (4096, 512), (512, 1), 0), view_244, out=buf53)
        buf530 = empty((64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf523, (64, 512, 64), (64, 4096, 1), 0), permute_520, out=buf530)
        del permute_520
        buf532 = empty((1, 64, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_12.run(buf530, alias_51, buf532, 32768, 512, grid=grid(32768), stream=stream0)
        del alias_51
        buf533 = reinterpret_tensor(buf523, (64, 64, 512), (32768, 512, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_521, reinterpret_tensor(buf532, (64, 512, 512), (262144, 512, 1), 0), out=buf533)
        del permute_521
        buf138 = empty((1, 4096), device='cuda', dtype=torch.float32)
        buf403 = reinterpret_tensor(buf138, (4096, ), (1, ), 0); del buf138  # reuse
        buf545 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_21.run(buf545, buf45, buf87, buf129, buf171, buf213, buf255, buf439, buf297, buf481, buf339, buf533, buf391, 4096, 512, grid=grid(4096), stream=stream0)
        buf57 = reinterpret_tensor(buf530, (4096, 4096), (4096, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (4096, 512), (1, 4096), 0), view_244, out=buf57)
        del view_244
        buf58 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf55, buf58, 16384, 128, grid=grid(16384), stream=stream0)
        buf100 = buf538; del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf97, buf100, 16384, 128, grid=grid(16384), stream=stream0)
        buf142 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf139, buf142, 16384, 128, grid=grid(16384), stream=stream0)
        buf184 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf181, buf184, 16384, 128, grid=grid(16384), stream=stream0)
        buf226 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf223, buf226, 16384, 128, grid=grid(16384), stream=stream0)
        buf268 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf265, buf268, 16384, 128, grid=grid(16384), stream=stream0)
        buf310 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf307, buf310, 16384, 128, grid=grid(16384), stream=stream0)
        buf352 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf349, buf352, 16384, 128, grid=grid(16384), stream=stream0)
        buf408 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf405, buf408, 16384, 128, grid=grid(16384), stream=stream0)
        buf452 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf449, buf452, 16384, 128, grid=grid(16384), stream=stream0)
        buf494 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf491, buf494, 16384, 128, grid=grid(16384), stream=stream0)
        buf534 = reinterpret_tensor(buf55, (64, 512, 64), (32768, 64, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf532, (64, 512, 512), (262144, 512, 1), 0), permute_522, out=buf534)
        del permute_522
        buf547 = reinterpret_tensor(buf45, (512, 4096), (4096, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf534, buf547, 2097152, grid=grid(2097152), stream=stream0)
        del buf534
        buf550 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf547, buf550, 16384, 128, grid=grid(16384), stream=stream0)
        buf101 = empty((1, 4096), device='cuda', dtype=torch.float32)
        buf410 = reinterpret_tensor(buf101, (4096, ), (1, ), 0); del buf101  # reuse
        buf552 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.sum]
        triton_per_fused_add_sum_17.run(buf552, buf58, buf100, buf142, buf184, buf226, buf268, buf452, buf310, buf494, buf352, buf550, buf408, 4096, 4, grid=grid(4096), stream=stream0)
        del buf100
        del buf142
        del buf184
        del buf226
        del buf268
        del buf310
        del buf352
        del buf408
        del buf452
        del buf494
        del buf550
        buf67 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (4096, 512), (1, 4096), 0), view_242, out=buf67)
        del buf63
        del view_242
        buf72 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (16384, 512), (1, 16384), 0), view_240, out=buf72)
        del buf70
        del view_240
        buf80 = reinterpret_tensor(buf532, (4096, 4096), (4096, 1), 0); del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (4096, 512), (1, 4096), 0), view_238, out=buf80)
        del buf76
        del view_238
        buf91 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (4096, 512), (1, 4096), 0), view_222, out=buf91)
        del buf89
        buf95 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (4096, 512), (512, 1), 0), view_222, out=buf95)
        del buf87
        buf99 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (4096, 512), (1, 4096), 0), view_222, out=buf99)
        del buf97
        del view_222
        buf109 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (4096, 512), (1, 4096), 0), view_220, out=buf109)
        del buf105
        del view_220
        buf114 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (16384, 512), (1, 16384), 0), view_218, out=buf114)
        del buf112
        del view_218
        buf122 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf118, (4096, 512), (1, 4096), 0), view_216, out=buf122)
        del buf118
        del view_216
        buf133 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (4096, 512), (1, 4096), 0), view_200, out=buf133)
        del buf131
        buf137 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (4096, 512), (512, 1), 0), view_200, out=buf137)
        del buf129
        buf141 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (4096, 512), (1, 4096), 0), view_200, out=buf141)
        del buf139
        del view_200
        buf151 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (4096, 512), (1, 4096), 0), view_198, out=buf151)
        del buf147
        del view_198
        buf156 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (16384, 512), (1, 16384), 0), view_196, out=buf156)
        del buf154
        del view_196
        buf164 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (4096, 512), (1, 4096), 0), view_194, out=buf164)
        del buf160
        del view_194
        buf175 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (4096, 512), (1, 4096), 0), view_178, out=buf175)
        del buf173
        buf179 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 512), (512, 1), 0), view_178, out=buf179)
        del buf171
        buf183 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (4096, 512), (1, 4096), 0), view_178, out=buf183)
        del buf181
        del view_178
        buf193 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (4096, 512), (1, 4096), 0), view_176, out=buf193)
        del buf189
        del view_176
        buf198 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (16384, 512), (1, 16384), 0), view_174, out=buf198)
        del buf196
        del view_174
        buf206 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (4096, 512), (1, 4096), 0), view_172, out=buf206)
        del buf202
        del view_172
        buf217 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (4096, 512), (1, 4096), 0), view_156, out=buf217)
        del buf215
        buf221 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (4096, 512), (512, 1), 0), view_156, out=buf221)
        del buf213
        buf225 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (4096, 512), (1, 4096), 0), view_156, out=buf225)
        del buf223
        del view_156
        buf235 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (4096, 512), (1, 4096), 0), view_154, out=buf235)
        del buf231
        del view_154
        buf240 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (16384, 512), (1, 16384), 0), view_152, out=buf240)
        del buf238
        del view_152
        buf248 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (4096, 512), (1, 4096), 0), view_150, out=buf248)
        del buf244
        del view_150
        buf259 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (4096, 512), (1, 4096), 0), view_134, out=buf259)
        del buf257
        buf263 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (4096, 512), (512, 1), 0), view_134, out=buf263)
        del buf255
        buf267 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (4096, 512), (1, 4096), 0), view_134, out=buf267)
        del buf265
        del view_134
        buf277 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (4096, 512), (1, 4096), 0), view_132, out=buf277)
        del buf273
        del view_132
        buf282 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (16384, 512), (1, 16384), 0), view_130, out=buf282)
        del buf280
        del view_130
        buf290 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (4096, 512), (1, 4096), 0), view_128, out=buf290)
        del buf286
        del view_128
        buf301 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (4096, 512), (1, 4096), 0), view_112, out=buf301)
        del buf299
        buf305 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (4096, 512), (512, 1), 0), view_112, out=buf305)
        del buf297
        buf309 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (4096, 512), (1, 4096), 0), view_112, out=buf309)
        del buf307
        del view_112
        buf319 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (4096, 512), (1, 4096), 0), view_110, out=buf319)
        del buf315
        del view_110
        buf324 = empty((16384, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (16384, 512), (1, 16384), 0), view_108, out=buf324)
        del buf322
        del view_108
        buf332 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf328, (4096, 512), (1, 4096), 0), view_106, out=buf332)
        del buf328
        del view_106
        buf343 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (4096, 512), (1, 4096), 0), view_90, out=buf343)
        del buf341
        buf347 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (4096, 512), (512, 1), 0), view_90, out=buf347)
        del buf339
        buf351 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (4096, 512), (1, 4096), 0), view_90, out=buf351)
        del buf349
        del view_90
        buf363 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (4096, 512), (1, 4096), 0), view_88, out=buf363)
        del buf357
        del view_88
        buf419 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (4096, 512), (1, 4096), 0), view_66, out=buf419)
        del buf415
        del view_66
        buf461 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (4096, 512), (1, 4096), 0), view_44, out=buf461)
        del buf457
        del view_44
        buf505 = empty((4096, 16384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (4096, 512), (1, 4096), 0), view_22, out=buf505)
        del buf499
        del view_22
        buf367 = buf109; del buf109  # reuse
        buf509 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf509, buf25, buf67, buf151, buf193, buf235, buf277, buf319, buf363, buf419, buf461, buf505, 67108864, grid=grid(67108864), stream=stream0)
        del buf151
        del buf193
        del buf235
        del buf25
        del buf277
        del buf319
        del buf363
        buf370 = reinterpret_tensor(buf67, (16384, 4096), (4096, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (16384, 512), (1, 16384), 0), view_86, out=buf370)
        del buf368
        del view_86
        buf424 = reinterpret_tensor(buf505, (16384, 4096), (4096, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (16384, 512), (1, 16384), 0), view_64, out=buf424)
        del buf422
        del view_64
        buf466 = reinterpret_tensor(buf461, (16384, 4096), (4096, 1), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (16384, 512), (1, 16384), 0), view_42, out=buf466)
        del buf464
        del view_42
        buf512 = reinterpret_tensor(buf419, (16384, 4096), (4096, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf510, (16384, 512), (1, 16384), 0), view_20, out=buf512)
        del buf510
        del view_20
        buf373 = buf114; del buf114  # reuse
        buf515 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf515, buf30, buf72, buf156, buf198, buf240, buf282, buf324, buf370, buf424, buf466, buf512, 67108864, grid=grid(67108864), stream=stream0)
        del buf156
        del buf198
        del buf240
        del buf282
        del buf30
        del buf324
        del buf370
        del buf424
        del buf466
        del buf512
        del buf72
        buf382 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (4096, 512), (1, 4096), 0), view_84, out=buf382)
        del buf376
        del view_84
        buf432 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (4096, 512), (1, 4096), 0), view_62, out=buf432)
        del buf428
        del view_62
        buf474 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf470, (4096, 512), (1, 4096), 0), view_40, out=buf474)
        del buf470
        del view_40
        buf524 = empty((4096, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (4096, 512), (1, 4096), 0), view_18, out=buf524)
        del view_18
        buf386 = buf122; del buf122  # reuse
        buf528 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf528, buf38, buf80, buf164, buf206, buf248, buf290, buf332, buf382, buf432, buf474, buf524, 16777216, grid=grid(16777216), stream=stream0)
        del buf164
        del buf206
        del buf248
        del buf290
        del buf332
        del buf38
        del buf382
        buf395 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf393, (4096, 512), (1, 4096), 0), view_68, out=buf395)
        del buf393
        buf443 = buf524; del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf441, (4096, 512), (1, 4096), 0), view_46, out=buf443)
        del buf441
        buf485 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf483, (4096, 512), (1, 4096), 0), view_24, out=buf485)
        del buf483
        buf537 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (4096, 512), (1, 4096), 0), view_2, out=buf537)
        buf399 = buf133; del buf133  # reuse
        buf541 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf541, buf49, buf91, buf175, buf217, buf259, buf301, buf343, buf395, buf443, buf485, buf537, 16777216, grid=grid(16777216), stream=stream0)
        del buf175
        del buf217
        del buf259
        del buf301
        del buf343
        del buf395
        del buf443
        buf401 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (4096, 512), (512, 1), 0), view_68, out=buf401)
        del buf391
        buf447 = buf537; del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (4096, 512), (512, 1), 0), view_46, out=buf447)
        del buf439
        buf489 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf481, (4096, 512), (512, 1), 0), view_24, out=buf489)
        del buf481
        buf543 = buf485; del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (4096, 512), (512, 1), 0), view_2, out=buf543)
        buf404 = buf137; del buf137  # reuse
        buf546 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf546, buf53, buf95, buf179, buf221, buf263, buf305, buf347, buf401, buf447, buf489, buf543, 16777216, grid=grid(16777216), stream=stream0)
        del buf179
        del buf221
        del buf263
        del buf305
        del buf347
        del buf401
        del buf447
        buf407 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (4096, 512), (1, 4096), 0), view_68, out=buf407)
        del buf405
        del view_68
        buf451 = buf543; del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (4096, 512), (1, 4096), 0), view_46, out=buf451)
        del buf449
        del view_46
        buf493 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf491, (4096, 512), (1, 4096), 0), view_24, out=buf493)
        del view_24
        buf549 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf547, (4096, 512), (1, 4096), 0), view_2, out=buf549)
        del view_2
        buf411 = buf141; del buf141  # reuse
        buf553 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf553, buf57, buf99, buf183, buf225, buf267, buf309, buf351, buf407, buf451, buf493, buf549, 16777216, grid=grid(16777216), stream=stream0)
        del buf183
        del buf225
        del buf267
        del buf309
        del buf351
        del buf407
        del buf451
        del buf493
        del buf549
        del buf57
        del buf99
        buf536 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf535, permute_164, out=buf536)
        del permute_164
        buf542 = buf535; del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (512, 4096), (1, 512), 0), permute_168, out=buf542)
        del permute_168
        buf548 = reinterpret_tensor(buf533, (512, 4096), (4096, 1), 0); del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf547, permute_172, out=buf548)
        del buf547
        del permute_172
        buf554 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_24.run(buf554, buf536, buf542, buf548, 2097152, grid=grid(2097152), stream=stream0)
        del buf536
        del buf542
        del buf548
        buf555 = reinterpret_tensor(buf14, (512, 128), (128, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf554, (512, 4096), (4096, 1), 0), permute_539, out=buf555)
        del permute_539
        buf556 = empty((4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf554, (4096, 512), (1, 4096), 0), view, out=buf556)
        del view
        buf557 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf554, buf557, 16384, 128, grid=grid(16384), stream=stream0)
        del buf554
        buf558 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf557, buf558, 4096, 4, grid=grid(4096), stream=stream0)
        del buf557
        buf565 = reinterpret_tensor(buf6, (1, 512, 128), (65536, 128, 1), 0); del buf6  # reuse
        buf569 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf573 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_26.run(buf555, primals_4, mul_1, div_63, slice_2, expand, primals_32, buf565, buf569, buf573, 512, 128, grid=grid(512), stream=stream0)
        del div_63
        del primals_4
        buf562 = empty((128, ), device='cuda', dtype=torch.float32)
        buf563 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf555, mul_1, buf562, buf563, 128, 512, grid=grid(128), stream=stream0)
        del mul_1
        buf564 = buf555; del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_28.run(buf564, 65536, grid=grid(65536), stream=stream0)
        aten.index_put_(buf564, [slice_2], buf565, True)
        del buf565
        del slice_2
        buf568 = empty((2, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_29.run(buf568, 256, grid=grid(256), stream=stream0)
        aten.index_put_(buf568, [expand], buf569, True)
        del buf569
        del expand
        buf572 = empty((30000, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_30.run(buf572, 3840000, grid=grid(3840000), stream=stream0)
        aten.index_put_(buf572, [primals_32], buf573, True)
        del buf573
        del primals_32
        return (buf572, buf568, buf564, buf562, buf563, reinterpret_tensor(buf556, (4096, 128), (128, 1), 0), reinterpret_tensor(buf558, (4096, ), (1, ), 0), buf553, buf552, buf546, buf545, buf541, buf540, buf528, buf527, buf521, buf522, buf515, buf514, buf509, buf508, buf502, buf503, reinterpret_tensor(buf16, (128, 4096), (4096, 1), 0), reinterpret_tensor(buf18, (128, ), (1, ), 0), buf12, buf13, reinterpret_tensor(buf7, (30000, 128), (128, 1), 0), reinterpret_tensor(buf8, (30000, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_33 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
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
    addmm_73 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    tanh_12 = rand_strided((1, 512, 128), (65536, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_25 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_268 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    sub_40 = rand_strided((512, 30000), (30000, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_135 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_139 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_143 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    permute_147 = rand_strided((16384, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_158 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    permute_164 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_168 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_172 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_192 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_225 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_324 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_356 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_357 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_390 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_420 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_422 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_423 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_454 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_456 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_49 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_488 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_489 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_519 = rand_strided((64, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_520 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    alias_51 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((64, 64, 512), (64, 1, 4096), device='cuda:0', dtype=torch.float32)
    permute_522 = rand_strided((64, 512, 64), (64, 4096, 1), device='cuda:0', dtype=torch.float32)
    permute_539 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 30000), (15360000, 30000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_16, primals_22, primals_26, primals_32, primals_33, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, addmm_73, tanh_12, getitem_51, rsqrt_25, view_268, sub_40, convert_element_type, permute_135, permute_139, div_27, permute_143, permute_147, div_28, permute_151, permute_156, permute_157, alias_29, permute_158, permute_159, permute_164, permute_168, permute_172, div_30, div_31, permute_189, permute_190, alias_31, permute_191, permute_192, div_33, div_34, permute_222, permute_223, alias_33, permute_224, permute_225, div_36, div_37, permute_255, permute_256, alias_35, permute_257, permute_258, div_39, div_40, permute_288, permute_289, alias_37, permute_290, permute_291, div_42, div_43, permute_321, permute_322, alias_39, permute_323, permute_324, div_45, div_46, permute_354, permute_355, alias_41, permute_356, permute_357, div_48, div_49, permute_387, permute_388, alias_43, permute_389, permute_390, div_51, div_52, permute_420, permute_421, alias_45, permute_422, permute_423, div_54, div_55, permute_453, permute_454, alias_47, permute_455, permute_456, div_57, div_58, permute_486, permute_487, alias_49, permute_488, permute_489, div_60, div_61, permute_519, permute_520, alias_51, permute_521, permute_522, permute_539, div_63, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AlbertForMaskedLM', benchmark_compiled_module)
