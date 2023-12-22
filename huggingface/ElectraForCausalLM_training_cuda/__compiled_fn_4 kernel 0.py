
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


# kernel path: /tmp/torchinductor_youkaichao/tl/ctlulnxo6yl3bo5mk532y4uio6fpqnswn7pzvdpgnyfulxczxpit.py
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
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15596742
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


# kernel path: /tmp/torchinductor_youkaichao/6r/c6rzvuwcqbnvkgdgnziodpypolopwwblpz7udo7gaop74xzjeci2.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 511
    rnumel = 30522
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp4 = tl.load(in_ptr3 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp3 / tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp1, tmp6, tmp7)
        tmp9 = tmp0 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c252eaa3sr75o2jo4ukj6vsyypqanrrokydewtzmnhzrwgslzsah.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15627264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 30522)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp8 = tl.load(in_ptr4 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp1 = x1
    tmp2 = tl.full([1], 511, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x2), tmp3 & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (x1), tmp3 & xmask, eviction_policy='evict_last').to(tl.int1)
    tmp10 = tmp7 / tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp4 * tmp12
    tmp14 = tl.load(in_ptr5 + (x2), tmp3 & xmask, other=0.0)
    tmp15 = tl.exp(tmp14)
    tmp16 = tl.load(in_ptr6 + (x1), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp13 - tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp3, tmp18, tmp19)
    tmp21 = tl.where(tmp3, tmp20, tmp11)
    tmp22 = tmp0 + tmp21
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4nubqp3xa4ostxpcquwfuojeatw5pne3r6zcwtgsnmckk6kput.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30522
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
        tmp0 = tl.load(in_ptr0 + (x0 + (30522*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czwteadvr56smtaw3g4qwnzlwvek74lh4vpybgde4ru5v55x6mnk.py
# Source Nodes: [hidden_states_111], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
# hidden_states_111 => add_100, erf_12, mul_88
triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp20 = tl.load(in_ptr4 + (r1 + (128*x0)), rmask & xmask, other=0.0)
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
    tmp21 = 0.7071067811865476
    tmp22 = tmp20 * tmp21
    tmp23 = tl.math.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = 0.5
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 * tmp20
    tmp29 = -0.5
    tmp30 = tmp28 * tmp29
    tmp31 = tl.exp(tmp30)
    tmp32 = 0.3989422804014327
    tmp33 = tmp31 * tmp32
    tmp34 = tmp20 * tmp33
    tmp35 = tmp27 + tmp34
    tmp36 = tmp19 * tmp35
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp36, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfv2f7i6bgj5a7le5ah4zs7eam5ei25m4jv4ayrhmro75pfueved.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6bzechqtsftk6emgo6jhgeacvdtwwsd5j6vo45xh5lf5entk53.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/x5/cx57mqmxioa3vknv33s7wjzzpqsyfmzirg4rgwirt42lp4isi46p.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gc/cgc46u4zeqbsqxyfxjdhuci6ajcrwjkzst5ka5dgvsqyaiowverd.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_10', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/77/c774mo6lroqn2yy4mlvwruuhuyw3ilm76s2blac55rowhliwiyq4.py
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
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_11', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wq/cwq7ixnmuuenzzlawrpcbpetnkucwdijoqigcnpr5bl357abg7c4.py
# Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_11 => add_96, erf_11, mul_83
triton_poi_fused_gelu_gelu_backward_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_12', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/wf/cwfmrxf7nkxltzzynvv7o4c5jgecaqadi5fm6njr7xnhjdxetonv.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_13', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/e6/ce65jz66hggl7w3mbqm2o7lwsnfr6yh4x5xg37p34fvlekjsdbgq.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_14', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3t4a6m45mycoatx63ejcwmjjz6mnwb4z2mtty7jgtkwpjtunyw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/is/cish6zq66bqyjq7aezdh77hl3hxl7ygglo2uinpfeefivh6yz4cg.py
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
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_16', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hh/chhy3xus3wiylafe37rsrccwbdfokq4sb36rbnrsc3vghfd3rore.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/je/cje6yrujhg7jxhgubvdx2dswrp7wch7ttl5iiz623wasgpnau2ux.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_18', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/fo/cfo5zi42hkmr6nhrqopnh4fvlslldjarnygja5j3s6sfx2vlshes.py
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_19', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/3f/c3feoaflfutz4g4co2fy44zvnantf6bzckvanrqnocalwiqcuetp.py
# Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_20', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jq/cjq7rzsdedaiowindclttl2p4yma5td5tmv3nfqpegj7x3g4lozz.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_21', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ia/ciadtcufne6hon6ardlv4eqzfj3mcv3gxvfltpbll5vv3k4eep5y.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_22', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yw/cywq4nymdutu3u6k36f3azm4bwtf4nliyz6qp2xv4bir6tqbsrtz.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_23', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7v/c7v4u52rfgc4kyvmmtidnwy3tl2msbrxye3opcr3saikqcufcwmr.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_24', 'mutated_arg_names': []},
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
    primals_4, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, primals_112, primals_118, primals_128, primals_134, primals_144, primals_150, primals_160, primals_166, primals_176, primals_182, primals_192, primals_198, primals_202, primals_209, expand, slice_4, mul_1, getitem_3, view, view_2, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_18, getitem_7, mul_3, view_20, addmm_5, view_22, getitem_11, mul_8, view_24, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_40, getitem_17, mul_10, view_42, addmm_11, view_44, getitem_21, mul_15, view_46, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_62, getitem_27, mul_17, view_64, addmm_17, view_66, getitem_31, mul_22, view_68, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_84, getitem_37, mul_24, view_86, addmm_23, view_88, getitem_41, mul_29, view_90, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_106, getitem_47, mul_31, view_108, addmm_29, view_110, getitem_51, mul_36, view_112, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_128, getitem_57, mul_38, view_130, addmm_35, view_132, getitem_61, mul_43, view_134, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_150, getitem_67, mul_45, view_152, addmm_41, view_154, getitem_71, mul_50, view_156, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_172, getitem_77, mul_52, view_174, addmm_47, view_176, getitem_81, mul_57, view_178, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_194, getitem_87, mul_59, view_196, addmm_53, view_198, getitem_91, mul_64, view_200, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_216, getitem_97, mul_66, view_218, addmm_59, view_220, getitem_101, mul_71, view_222, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_238, getitem_107, mul_73, view_240, addmm_65, view_242, getitem_111, mul_78, view_244, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_260, getitem_117, mul_80, view_262, addmm_71, view_264, getitem_121, mul_85, view_266, addmm_73, mul_90, view_268, sub_40, convert_element_type, ne_3, where_2, permute_135, div_26, permute_139, div_27, permute_143, permute_147, div_28, permute_151, permute_163, permute_168, permute_172, div_30, permute_176, permute_180, div_31, permute_184, permute_196, permute_201, permute_205, div_33, permute_209, permute_213, div_34, permute_217, permute_229, permute_234, permute_238, div_36, permute_242, permute_246, div_37, permute_250, permute_262, permute_267, permute_271, div_39, permute_275, permute_279, div_40, permute_283, permute_295, permute_300, permute_304, div_42, permute_308, permute_312, div_43, permute_316, permute_328, permute_333, permute_337, div_45, permute_341, permute_345, div_46, permute_349, permute_361, permute_366, permute_370, div_48, permute_374, permute_378, div_49, permute_382, permute_394, permute_399, permute_403, div_51, permute_407, permute_411, div_52, permute_415, permute_427, permute_432, permute_436, div_54, permute_440, permute_444, div_55, permute_448, permute_460, permute_465, permute_469, div_57, permute_473, permute_477, div_58, permute_481, permute_493, permute_498, permute_502, div_60, permute_506, permute_510, div_61, permute_514, permute_526, permute_531, permute_535, permute_539, div_63, tangents_1, tangents_2 = args
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
    assert_size_stride(primals_202, (128, ), (1, ))
    assert_size_stride(primals_209, (1, 512), (512, 1))
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
    assert_size_stride(addmm_73, (512, 128), (128, 1))
    assert_size_stride(mul_90, (1, 512, 128), (65536, 128, 1))
    assert_size_stride(view_268, (512, 128), (128, 1))
    assert_size_stride(sub_40, (511, 30522), (30522, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(ne_3, (511, 1), (1, 1))
    assert_size_stride(where_2, (511, 1), (1, 1))
    assert_size_stride(permute_135, (30522, 128), (128, 1))
    assert_size_stride(div_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_139, (128, 256), (256, 1))
    assert_size_stride(div_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_143, (256, 1024), (1024, 1))
    assert_size_stride(permute_147, (1024, 256), (256, 1))
    assert_size_stride(div_28, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_151, (256, 256), (256, 1))
    assert_size_stride(permute_163, (256, 256), (256, 1))
    assert_size_stride(permute_168, (256, 256), (256, 1))
    assert_size_stride(permute_172, (256, 256), (256, 1))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_176, (256, 1024), (1024, 1))
    assert_size_stride(permute_180, (1024, 256), (256, 1))
    assert_size_stride(div_31, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_184, (256, 256), (256, 1))
    assert_size_stride(permute_196, (256, 256), (256, 1))
    assert_size_stride(permute_201, (256, 256), (256, 1))
    assert_size_stride(permute_205, (256, 256), (256, 1))
    assert_size_stride(div_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_209, (256, 1024), (1024, 1))
    assert_size_stride(permute_213, (1024, 256), (256, 1))
    assert_size_stride(div_34, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_217, (256, 256), (256, 1))
    assert_size_stride(permute_229, (256, 256), (256, 1))
    assert_size_stride(permute_234, (256, 256), (256, 1))
    assert_size_stride(permute_238, (256, 256), (256, 1))
    assert_size_stride(div_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_242, (256, 1024), (1024, 1))
    assert_size_stride(permute_246, (1024, 256), (256, 1))
    assert_size_stride(div_37, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_250, (256, 256), (256, 1))
    assert_size_stride(permute_262, (256, 256), (256, 1))
    assert_size_stride(permute_267, (256, 256), (256, 1))
    assert_size_stride(permute_271, (256, 256), (256, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_275, (256, 1024), (1024, 1))
    assert_size_stride(permute_279, (1024, 256), (256, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_283, (256, 256), (256, 1))
    assert_size_stride(permute_295, (256, 256), (256, 1))
    assert_size_stride(permute_300, (256, 256), (256, 1))
    assert_size_stride(permute_304, (256, 256), (256, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_308, (256, 1024), (1024, 1))
    assert_size_stride(permute_312, (1024, 256), (256, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_316, (256, 256), (256, 1))
    assert_size_stride(permute_328, (256, 256), (256, 1))
    assert_size_stride(permute_333, (256, 256), (256, 1))
    assert_size_stride(permute_337, (256, 256), (256, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_341, (256, 1024), (1024, 1))
    assert_size_stride(permute_345, (1024, 256), (256, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_349, (256, 256), (256, 1))
    assert_size_stride(permute_361, (256, 256), (256, 1))
    assert_size_stride(permute_366, (256, 256), (256, 1))
    assert_size_stride(permute_370, (256, 256), (256, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_374, (256, 1024), (1024, 1))
    assert_size_stride(permute_378, (1024, 256), (256, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_382, (256, 256), (256, 1))
    assert_size_stride(permute_394, (256, 256), (256, 1))
    assert_size_stride(permute_399, (256, 256), (256, 1))
    assert_size_stride(permute_403, (256, 256), (256, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_407, (256, 1024), (1024, 1))
    assert_size_stride(permute_411, (1024, 256), (256, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_415, (256, 256), (256, 1))
    assert_size_stride(permute_427, (256, 256), (256, 1))
    assert_size_stride(permute_432, (256, 256), (256, 1))
    assert_size_stride(permute_436, (256, 256), (256, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_440, (256, 1024), (1024, 1))
    assert_size_stride(permute_444, (1024, 256), (256, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_448, (256, 256), (256, 1))
    assert_size_stride(permute_460, (256, 256), (256, 1))
    assert_size_stride(permute_465, (256, 256), (256, 1))
    assert_size_stride(permute_469, (256, 256), (256, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_473, (256, 1024), (1024, 1))
    assert_size_stride(permute_477, (1024, 256), (256, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_481, (256, 256), (256, 1))
    assert_size_stride(permute_493, (256, 256), (256, 1))
    assert_size_stride(permute_498, (256, 256), (256, 1))
    assert_size_stride(permute_502, (256, 256), (256, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_506, (256, 1024), (1024, 1))
    assert_size_stride(permute_510, (1024, 256), (256, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_514, (256, 256), (256, 1))
    assert_size_stride(permute_526, (256, 256), (256, 1))
    assert_size_stride(permute_531, (256, 256), (256, 1))
    assert_size_stride(permute_535, (256, 256), (256, 1))
    assert_size_stride(permute_539, (256, 128), (128, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 30522), (15627264, 30522, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((511, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_0.run(buf0, 15596742, grid=grid(15596742), stream=stream0)
        aten.scatter_(buf0,1,where_2,-1.0)
        del where_2
        buf3 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1.run(buf0, ne_3, tangents_1, convert_element_type, buf3, 511, 30522, grid=grid(511), stream=stream0)
        buf4 = empty((1, 512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
        triton_poi_fused_add_slice_backward_2.run(tangents_2, buf0, ne_3, tangents_1, convert_element_type, sub_40, buf3, buf4, 15627264, grid=grid(15627264), stream=stream0)
        del buf0
        del buf3
        del convert_element_type
        del ne_3
        del sub_40
        del tangents_1
        del tangents_2
        buf5 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 30522), (30522, 1), 0), permute_135, out=buf5)
        del permute_135
        buf6 = empty((30522, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (30522, 512), (1, 30522), 0), view_268, out=buf6)
        del view_268
        buf7 = empty((1, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf4, buf7, 30522, 512, grid=grid(30522), stream=stream0)
        del buf4
        buf12 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_111], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4.run(buf5, primals_202, mul_90, div_26, addmm_73, buf12, 512, 128, grid=grid(512), stream=stream0)
        del addmm_73
        del div_26
        del primals_202
        buf10 = empty((128, ), device='cuda', dtype=torch.float32)
        buf11 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf5, mul_90, buf10, buf11, 128, 512, grid=grid(128), stream=stream0)
        del mul_90
        buf13 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (512, 128), (128, 1), 0), permute_139, out=buf13)
        del permute_139
        buf14 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (128, 512), (1, 128), 0), view_266, out=buf14)
        del view_266
        buf15 = empty_strided((1, 128, 4), (512, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf12, buf15, 512, 128, grid=grid(512), stream=stream0)
        buf16 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf15, buf16, 128, 4, grid=grid(128), stream=stream0)
        del buf15
        buf19 = empty((1, 512, 256), device='cuda', dtype=torch.float32)
        buf22 = empty((1, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_8.run(buf13, primals_198, mul_85, div_27, getitem_121, buf19, buf22, 512, 256, grid=grid(512), stream=stream0)
        del div_27
        del getitem_121
        del primals_198
        buf20 = empty((256, ), device='cuda', dtype=torch.float32)
        buf21 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf13, mul_85, buf20, buf21, 256, 512, grid=grid(256), stream=stream0)
        del mul_85
        buf23 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (512, 256), (256, 1), 0), permute_143, out=buf23)
        del permute_143
        buf24 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (256, 512), (1, 256), 0), view_264, out=buf24)
        del view_264
        buf25 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf22, buf25, 1024, 128, grid=grid(1024), stream=stream0)
        buf26 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf25, buf26, 256, 4, grid=grid(256), stream=stream0)
        buf27 = reinterpret_tensor(buf23, (1, 512, 1024), (524288, 1024, 1), 0); del buf23  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf27, addmm_71, 524288, grid=grid(524288), stream=stream0)
        del addmm_71
        buf28 = reinterpret_tensor(buf22, (512, 256), (256, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 1024), (1024, 1), 0), permute_147, out=buf28)
        del permute_147
        buf29 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (1024, 512), (1, 1024), 0), view_262, out=buf29)
        del view_262
        buf30 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf27, buf30, 4096, 128, grid=grid(4096), stream=stream0)
        buf31 = reinterpret_tensor(buf25, (1, 1024), (1024, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf30, buf31, 1024, 4, grid=grid(1024), stream=stream0)
        buf34 = reinterpret_tensor(buf13, (1, 512, 256), (131072, 256, 1), 0); del buf13  # reuse
        buf37 = empty((1, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf19, buf28, primals_192, mul_80, div_28, getitem_117, buf34, buf37, 512, 256, grid=grid(512), stream=stream0)
        del div_28
        del getitem_117
        del primals_192
        buf35 = empty((256, ), device='cuda', dtype=torch.float32)
        buf36 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf19, buf28, mul_80, buf35, buf36, 256, 512, grid=grid(256), stream=stream0)
        del buf19
        del mul_80
        buf38 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (512, 256), (256, 1), 0), permute_151, out=buf38)
        del permute_151
        buf39 = reinterpret_tensor(buf12, (256, 256), (256, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (256, 512), (1, 256), 0), view_260, out=buf39)
        del view_260
        buf40 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf37, buf40, 1024, 128, grid=grid(1024), stream=stream0)
        buf41 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf40, buf41, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf42 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf38, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_1
        del clone_default
        del clone_default_1
        del clone_default_2
        del getitem_127
        del getitem_128
        del getitem_129
        buf43 = buf42[0]
        buf44 = buf42[1]
        buf45 = buf42[2]
        del buf42
        buf46 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (512, 256), (256, 1), 0), permute_163, out=buf46)
        del permute_163
        buf47 = reinterpret_tensor(buf5, (256, 256), (256, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (256, 512), (1, 256), 0), view_244, out=buf47)
        buf48 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf45, buf48, 1024, 128, grid=grid(1024), stream=stream0)
        buf49 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf48, buf49, 256, 4, grid=grid(256), stream=stream0)
        buf50 = reinterpret_tensor(buf45, (512, 256), (256, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (512, 256), (256, 1), 0), permute_168, out=buf50)
        del permute_168
        buf51 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (256, 512), (1, 256), 0), view_244, out=buf51)
        buf52 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf44, buf52, 1024, 128, grid=grid(1024), stream=stream0)
        buf53 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf52, buf53, 256, 4, grid=grid(256), stream=stream0)
        buf54 = reinterpret_tensor(buf44, (512, 256), (256, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (512, 256), (256, 1), 0), permute_172, out=buf54)
        del permute_172
        buf55 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (256, 512), (1, 256), 0), view_244, out=buf55)
        del view_244
        buf56 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf43, buf56, 1024, 128, grid=grid(1024), stream=stream0)
        buf57 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf56, buf57, 256, 4, grid=grid(256), stream=stream0)
        buf61 = reinterpret_tensor(buf43, (1, 512, 256), (131072, 256, 1), 0); del buf43  # reuse
        buf64 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf34, buf46, buf50, buf54, primals_182, mul_78, div_30, getitem_111, buf61, buf64, 512, 256, grid=grid(512), stream=stream0)
        del div_30
        del getitem_111
        del primals_182
        buf62 = empty((256, ), device='cuda', dtype=torch.float32)
        buf63 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf34, buf46, buf50, buf54, mul_78, buf62, buf63, 256, 512, grid=grid(256), stream=stream0)
        del buf34
        del buf46
        del mul_78
        buf65 = reinterpret_tensor(buf27, (512, 1024), (1024, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 256), (256, 1), 0), permute_176, out=buf65)
        del permute_176
        buf66 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (256, 512), (1, 256), 0), view_242, out=buf66)
        del view_242
        buf67 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf64, buf67, 1024, 128, grid=grid(1024), stream=stream0)
        buf68 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf67, buf68, 256, 4, grid=grid(256), stream=stream0)
        buf69 = reinterpret_tensor(buf65, (1, 512, 1024), (524288, 1024, 1), 0); del buf65  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf69, addmm_65, 524288, grid=grid(524288), stream=stream0)
        del addmm_65
        buf70 = reinterpret_tensor(buf64, (512, 256), (256, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (512, 1024), (1024, 1), 0), permute_180, out=buf70)
        del permute_180
        buf71 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1024, 512), (1, 1024), 0), view_240, out=buf71)
        del view_240
        buf72 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf69, buf72, 4096, 128, grid=grid(4096), stream=stream0)
        buf73 = reinterpret_tensor(buf67, (1, 1024), (1024, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf72, buf73, 1024, 4, grid=grid(1024), stream=stream0)
        buf76 = reinterpret_tensor(buf54, (1, 512, 256), (131072, 256, 1), 0); del buf54  # reuse
        buf79 = reinterpret_tensor(buf50, (1, 512, 256), (131072, 256, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf61, buf70, primals_176, mul_73, div_31, getitem_107, buf76, buf79, 512, 256, grid=grid(512), stream=stream0)
        del div_31
        del getitem_107
        del primals_176
        buf77 = empty((256, ), device='cuda', dtype=torch.float32)
        buf78 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf61, buf70, mul_73, buf77, buf78, 256, 512, grid=grid(256), stream=stream0)
        del buf61
        del mul_73
        buf80 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (512, 256), (256, 1), 0), permute_184, out=buf80)
        del permute_184
        buf81 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (256, 512), (1, 256), 0), view_238, out=buf81)
        del view_238
        buf82 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf79, buf82, 1024, 128, grid=grid(1024), stream=stream0)
        buf83 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf82, buf83, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf84 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf80, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_134, getitem_135, getitem_136, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_3
        del clone_default_3
        del clone_default_4
        del clone_default_5
        del getitem_134
        del getitem_135
        del getitem_136
        buf85 = buf84[0]
        buf86 = buf84[1]
        buf87 = buf84[2]
        del buf84
        buf88 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (512, 256), (256, 1), 0), permute_196, out=buf88)
        del permute_196
        buf89 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (256, 512), (1, 256), 0), view_222, out=buf89)
        buf90 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf87, buf90, 1024, 128, grid=grid(1024), stream=stream0)
        buf91 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf90, buf91, 256, 4, grid=grid(256), stream=stream0)
        buf92 = reinterpret_tensor(buf87, (512, 256), (256, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (512, 256), (256, 1), 0), permute_201, out=buf92)
        del permute_201
        buf93 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (256, 512), (1, 256), 0), view_222, out=buf93)
        buf94 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf86, buf94, 1024, 128, grid=grid(1024), stream=stream0)
        buf95 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf94, buf95, 256, 4, grid=grid(256), stream=stream0)
        buf96 = reinterpret_tensor(buf86, (512, 256), (256, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (512, 256), (256, 1), 0), permute_205, out=buf96)
        del permute_205
        buf97 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (256, 512), (1, 256), 0), view_222, out=buf97)
        del view_222
        buf98 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf85, buf98, 1024, 128, grid=grid(1024), stream=stream0)
        buf99 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf98, buf99, 256, 4, grid=grid(256), stream=stream0)
        buf103 = reinterpret_tensor(buf85, (1, 512, 256), (131072, 256, 1), 0); del buf85  # reuse
        buf106 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf76, buf88, buf92, buf96, primals_166, mul_71, div_33, getitem_101, buf103, buf106, 512, 256, grid=grid(512), stream=stream0)
        del div_33
        del getitem_101
        del primals_166
        buf104 = empty((256, ), device='cuda', dtype=torch.float32)
        buf105 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf76, buf88, buf92, buf96, mul_71, buf104, buf105, 256, 512, grid=grid(256), stream=stream0)
        del buf76
        del buf88
        del mul_71
        buf107 = reinterpret_tensor(buf69, (512, 1024), (1024, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (512, 256), (256, 1), 0), permute_209, out=buf107)
        del permute_209
        buf108 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (256, 512), (1, 256), 0), view_220, out=buf108)
        del view_220
        buf109 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf106, buf109, 1024, 128, grid=grid(1024), stream=stream0)
        buf110 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf109, buf110, 256, 4, grid=grid(256), stream=stream0)
        buf111 = reinterpret_tensor(buf107, (1, 512, 1024), (524288, 1024, 1), 0); del buf107  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf111, addmm_59, 524288, grid=grid(524288), stream=stream0)
        del addmm_59
        buf112 = reinterpret_tensor(buf106, (512, 256), (256, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (512, 1024), (1024, 1), 0), permute_213, out=buf112)
        del permute_213
        buf113 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (1024, 512), (1, 1024), 0), view_218, out=buf113)
        del view_218
        buf114 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf111, buf114, 4096, 128, grid=grid(4096), stream=stream0)
        buf115 = reinterpret_tensor(buf109, (1, 1024), (1024, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf114, buf115, 1024, 4, grid=grid(1024), stream=stream0)
        buf118 = reinterpret_tensor(buf96, (1, 512, 256), (131072, 256, 1), 0); del buf96  # reuse
        buf121 = reinterpret_tensor(buf92, (1, 512, 256), (131072, 256, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf103, buf112, primals_160, mul_66, div_34, getitem_97, buf118, buf121, 512, 256, grid=grid(512), stream=stream0)
        del div_34
        del getitem_97
        del primals_160
        buf119 = empty((256, ), device='cuda', dtype=torch.float32)
        buf120 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf103, buf112, mul_66, buf119, buf120, 256, 512, grid=grid(256), stream=stream0)
        del buf103
        del mul_66
        buf122 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (512, 256), (256, 1), 0), permute_217, out=buf122)
        del permute_217
        buf123 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (256, 512), (1, 256), 0), view_216, out=buf123)
        del view_216
        buf124 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf121, buf124, 1024, 128, grid=grid(1024), stream=stream0)
        buf125 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf124, buf125, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf126 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf122, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_141, getitem_142, getitem_143, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_5
        del clone_default_6
        del clone_default_7
        del clone_default_8
        del getitem_141
        del getitem_142
        del getitem_143
        buf127 = buf126[0]
        buf128 = buf126[1]
        buf129 = buf126[2]
        del buf126
        buf130 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (512, 256), (256, 1), 0), permute_229, out=buf130)
        del permute_229
        buf131 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (256, 512), (1, 256), 0), view_200, out=buf131)
        buf132 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf129, buf132, 1024, 128, grid=grid(1024), stream=stream0)
        buf133 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf132, buf133, 256, 4, grid=grid(256), stream=stream0)
        buf134 = reinterpret_tensor(buf129, (512, 256), (256, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (512, 256), (256, 1), 0), permute_234, out=buf134)
        del permute_234
        buf135 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (256, 512), (1, 256), 0), view_200, out=buf135)
        buf136 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf128, buf136, 1024, 128, grid=grid(1024), stream=stream0)
        buf137 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf136, buf137, 256, 4, grid=grid(256), stream=stream0)
        buf138 = reinterpret_tensor(buf128, (512, 256), (256, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (512, 256), (256, 1), 0), permute_238, out=buf138)
        del permute_238
        buf139 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (256, 512), (1, 256), 0), view_200, out=buf139)
        del view_200
        buf140 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf127, buf140, 1024, 128, grid=grid(1024), stream=stream0)
        buf141 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf140, buf141, 256, 4, grid=grid(256), stream=stream0)
        buf145 = reinterpret_tensor(buf127, (1, 512, 256), (131072, 256, 1), 0); del buf127  # reuse
        buf148 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf118, buf130, buf134, buf138, primals_150, mul_64, div_36, getitem_91, buf145, buf148, 512, 256, grid=grid(512), stream=stream0)
        del div_36
        del getitem_91
        del primals_150
        buf146 = empty((256, ), device='cuda', dtype=torch.float32)
        buf147 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf118, buf130, buf134, buf138, mul_64, buf146, buf147, 256, 512, grid=grid(256), stream=stream0)
        del buf118
        del buf130
        del mul_64
        buf149 = reinterpret_tensor(buf111, (512, 1024), (1024, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 256), (256, 1), 0), permute_242, out=buf149)
        del permute_242
        buf150 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (256, 512), (1, 256), 0), view_198, out=buf150)
        del view_198
        buf151 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf148, buf151, 1024, 128, grid=grid(1024), stream=stream0)
        buf152 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf151, buf152, 256, 4, grid=grid(256), stream=stream0)
        buf153 = reinterpret_tensor(buf149, (1, 512, 1024), (524288, 1024, 1), 0); del buf149  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf153, addmm_53, 524288, grid=grid(524288), stream=stream0)
        del addmm_53
        buf154 = reinterpret_tensor(buf148, (512, 256), (256, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 1024), (1024, 1), 0), permute_246, out=buf154)
        del permute_246
        buf155 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (1024, 512), (1, 1024), 0), view_196, out=buf155)
        del view_196
        buf156 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf153, buf156, 4096, 128, grid=grid(4096), stream=stream0)
        buf157 = reinterpret_tensor(buf151, (1, 1024), (1024, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf156, buf157, 1024, 4, grid=grid(1024), stream=stream0)
        buf160 = reinterpret_tensor(buf138, (1, 512, 256), (131072, 256, 1), 0); del buf138  # reuse
        buf163 = reinterpret_tensor(buf134, (1, 512, 256), (131072, 256, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf145, buf154, primals_144, mul_59, div_37, getitem_87, buf160, buf163, 512, 256, grid=grid(512), stream=stream0)
        del div_37
        del getitem_87
        del primals_144
        buf161 = empty((256, ), device='cuda', dtype=torch.float32)
        buf162 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf145, buf154, mul_59, buf161, buf162, 256, 512, grid=grid(256), stream=stream0)
        del buf145
        del mul_59
        buf164 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (512, 256), (256, 1), 0), permute_250, out=buf164)
        del permute_250
        buf165 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (256, 512), (1, 256), 0), view_194, out=buf165)
        del view_194
        buf166 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf163, buf166, 1024, 128, grid=grid(1024), stream=stream0)
        buf167 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf166, buf167, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf168 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf164, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_148, getitem_149, getitem_150, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_7
        del clone_default_10
        del clone_default_11
        del clone_default_9
        del getitem_148
        del getitem_149
        del getitem_150
        buf169 = buf168[0]
        buf170 = buf168[1]
        buf171 = buf168[2]
        del buf168
        buf172 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (512, 256), (256, 1), 0), permute_262, out=buf172)
        del permute_262
        buf173 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (256, 512), (1, 256), 0), view_178, out=buf173)
        buf174 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf171, buf174, 1024, 128, grid=grid(1024), stream=stream0)
        buf175 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf174, buf175, 256, 4, grid=grid(256), stream=stream0)
        buf176 = reinterpret_tensor(buf171, (512, 256), (256, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (512, 256), (256, 1), 0), permute_267, out=buf176)
        del permute_267
        buf177 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (256, 512), (1, 256), 0), view_178, out=buf177)
        buf178 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf170, buf178, 1024, 128, grid=grid(1024), stream=stream0)
        buf179 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf178, buf179, 256, 4, grid=grid(256), stream=stream0)
        buf180 = reinterpret_tensor(buf170, (512, 256), (256, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (512, 256), (256, 1), 0), permute_271, out=buf180)
        del permute_271
        buf181 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (256, 512), (1, 256), 0), view_178, out=buf181)
        del view_178
        buf182 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf169, buf182, 1024, 128, grid=grid(1024), stream=stream0)
        buf183 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf182, buf183, 256, 4, grid=grid(256), stream=stream0)
        buf187 = reinterpret_tensor(buf169, (1, 512, 256), (131072, 256, 1), 0); del buf169  # reuse
        buf190 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf160, buf172, buf176, buf180, primals_134, mul_57, div_39, getitem_81, buf187, buf190, 512, 256, grid=grid(512), stream=stream0)
        del div_39
        del getitem_81
        del primals_134
        buf188 = empty((256, ), device='cuda', dtype=torch.float32)
        buf189 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf160, buf172, buf176, buf180, mul_57, buf188, buf189, 256, 512, grid=grid(256), stream=stream0)
        del buf160
        del buf172
        del mul_57
        buf191 = reinterpret_tensor(buf153, (512, 1024), (1024, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (512, 256), (256, 1), 0), permute_275, out=buf191)
        del permute_275
        buf192 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (256, 512), (1, 256), 0), view_176, out=buf192)
        del view_176
        buf193 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf190, buf193, 1024, 128, grid=grid(1024), stream=stream0)
        buf194 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf193, buf194, 256, 4, grid=grid(256), stream=stream0)
        buf195 = reinterpret_tensor(buf191, (1, 512, 1024), (524288, 1024, 1), 0); del buf191  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf195, addmm_47, 524288, grid=grid(524288), stream=stream0)
        del addmm_47
        buf196 = reinterpret_tensor(buf190, (512, 256), (256, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 1024), (1024, 1), 0), permute_279, out=buf196)
        del permute_279
        buf197 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (1024, 512), (1, 1024), 0), view_174, out=buf197)
        del view_174
        buf198 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf195, buf198, 4096, 128, grid=grid(4096), stream=stream0)
        buf199 = reinterpret_tensor(buf193, (1, 1024), (1024, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf198, buf199, 1024, 4, grid=grid(1024), stream=stream0)
        buf202 = reinterpret_tensor(buf180, (1, 512, 256), (131072, 256, 1), 0); del buf180  # reuse
        buf205 = reinterpret_tensor(buf176, (1, 512, 256), (131072, 256, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf187, buf196, primals_128, mul_52, div_40, getitem_77, buf202, buf205, 512, 256, grid=grid(512), stream=stream0)
        del div_40
        del getitem_77
        del primals_128
        buf203 = empty((256, ), device='cuda', dtype=torch.float32)
        buf204 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf187, buf196, mul_52, buf203, buf204, 256, 512, grid=grid(256), stream=stream0)
        del buf187
        del mul_52
        buf206 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (512, 256), (256, 1), 0), permute_283, out=buf206)
        del permute_283
        buf207 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (256, 512), (1, 256), 0), view_172, out=buf207)
        del view_172
        buf208 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf205, buf208, 1024, 128, grid=grid(1024), stream=stream0)
        buf209 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf208, buf209, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf210 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf206, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_155, getitem_156, getitem_157, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_9
        del clone_default_12
        del clone_default_13
        del clone_default_14
        del getitem_155
        del getitem_156
        del getitem_157
        buf211 = buf210[0]
        buf212 = buf210[1]
        buf213 = buf210[2]
        del buf210
        buf214 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (512, 256), (256, 1), 0), permute_295, out=buf214)
        del permute_295
        buf215 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (256, 512), (1, 256), 0), view_156, out=buf215)
        buf216 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf213, buf216, 1024, 128, grid=grid(1024), stream=stream0)
        buf217 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf216, buf217, 256, 4, grid=grid(256), stream=stream0)
        buf218 = reinterpret_tensor(buf213, (512, 256), (256, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 256), (256, 1), 0), permute_300, out=buf218)
        del permute_300
        buf219 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (256, 512), (1, 256), 0), view_156, out=buf219)
        buf220 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf212, buf220, 1024, 128, grid=grid(1024), stream=stream0)
        buf221 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf220, buf221, 256, 4, grid=grid(256), stream=stream0)
        buf222 = reinterpret_tensor(buf212, (512, 256), (256, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (512, 256), (256, 1), 0), permute_304, out=buf222)
        del permute_304
        buf223 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (256, 512), (1, 256), 0), view_156, out=buf223)
        del view_156
        buf224 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf211, buf224, 1024, 128, grid=grid(1024), stream=stream0)
        buf225 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf224, buf225, 256, 4, grid=grid(256), stream=stream0)
        buf229 = reinterpret_tensor(buf211, (1, 512, 256), (131072, 256, 1), 0); del buf211  # reuse
        buf232 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf202, buf214, buf218, buf222, primals_118, mul_50, div_42, getitem_71, buf229, buf232, 512, 256, grid=grid(512), stream=stream0)
        del div_42
        del getitem_71
        del primals_118
        buf230 = empty((256, ), device='cuda', dtype=torch.float32)
        buf231 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf202, buf214, buf218, buf222, mul_50, buf230, buf231, 256, 512, grid=grid(256), stream=stream0)
        del buf202
        del buf214
        del mul_50
        buf233 = reinterpret_tensor(buf195, (512, 1024), (1024, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (512, 256), (256, 1), 0), permute_308, out=buf233)
        del permute_308
        buf234 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (256, 512), (1, 256), 0), view_154, out=buf234)
        del view_154
        buf235 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf232, buf235, 1024, 128, grid=grid(1024), stream=stream0)
        buf236 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf235, buf236, 256, 4, grid=grid(256), stream=stream0)
        buf237 = reinterpret_tensor(buf233, (1, 512, 1024), (524288, 1024, 1), 0); del buf233  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf237, addmm_41, 524288, grid=grid(524288), stream=stream0)
        del addmm_41
        buf238 = reinterpret_tensor(buf232, (512, 256), (256, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (512, 1024), (1024, 1), 0), permute_312, out=buf238)
        del permute_312
        buf239 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (1024, 512), (1, 1024), 0), view_152, out=buf239)
        del view_152
        buf240 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf237, buf240, 4096, 128, grid=grid(4096), stream=stream0)
        buf241 = reinterpret_tensor(buf235, (1, 1024), (1024, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf240, buf241, 1024, 4, grid=grid(1024), stream=stream0)
        buf244 = reinterpret_tensor(buf222, (1, 512, 256), (131072, 256, 1), 0); del buf222  # reuse
        buf247 = reinterpret_tensor(buf218, (1, 512, 256), (131072, 256, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf229, buf238, primals_112, mul_45, div_43, getitem_67, buf244, buf247, 512, 256, grid=grid(512), stream=stream0)
        del div_43
        del getitem_67
        del primals_112
        buf245 = empty((256, ), device='cuda', dtype=torch.float32)
        buf246 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf229, buf238, mul_45, buf245, buf246, 256, 512, grid=grid(256), stream=stream0)
        del buf229
        del mul_45
        buf248 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (512, 256), (256, 1), 0), permute_316, out=buf248)
        del permute_316
        buf249 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (256, 512), (1, 256), 0), view_150, out=buf249)
        del view_150
        buf250 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf247, buf250, 1024, 128, grid=grid(1024), stream=stream0)
        buf251 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf250, buf251, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf252 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf248, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_162, getitem_163, getitem_164, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_11
        del clone_default_15
        del clone_default_16
        del clone_default_17
        del getitem_162
        del getitem_163
        del getitem_164
        buf253 = buf252[0]
        buf254 = buf252[1]
        buf255 = buf252[2]
        del buf252
        buf256 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (512, 256), (256, 1), 0), permute_328, out=buf256)
        del permute_328
        buf257 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (256, 512), (1, 256), 0), view_134, out=buf257)
        buf258 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf255, buf258, 1024, 128, grid=grid(1024), stream=stream0)
        buf259 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf258, buf259, 256, 4, grid=grid(256), stream=stream0)
        buf260 = reinterpret_tensor(buf255, (512, 256), (256, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (512, 256), (256, 1), 0), permute_333, out=buf260)
        del permute_333
        buf261 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (256, 512), (1, 256), 0), view_134, out=buf261)
        buf262 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf254, buf262, 1024, 128, grid=grid(1024), stream=stream0)
        buf263 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf262, buf263, 256, 4, grid=grid(256), stream=stream0)
        buf264 = reinterpret_tensor(buf254, (512, 256), (256, 1), 0); del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (512, 256), (256, 1), 0), permute_337, out=buf264)
        del permute_337
        buf265 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (256, 512), (1, 256), 0), view_134, out=buf265)
        del view_134
        buf266 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf253, buf266, 1024, 128, grid=grid(1024), stream=stream0)
        buf267 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf266, buf267, 256, 4, grid=grid(256), stream=stream0)
        buf271 = reinterpret_tensor(buf253, (1, 512, 256), (131072, 256, 1), 0); del buf253  # reuse
        buf274 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf244, buf256, buf260, buf264, primals_102, mul_43, div_45, getitem_61, buf271, buf274, 512, 256, grid=grid(512), stream=stream0)
        del div_45
        del getitem_61
        del primals_102
        buf272 = empty((256, ), device='cuda', dtype=torch.float32)
        buf273 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf244, buf256, buf260, buf264, mul_43, buf272, buf273, 256, 512, grid=grid(256), stream=stream0)
        del buf244
        del buf256
        del mul_43
        buf275 = reinterpret_tensor(buf237, (512, 1024), (1024, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (512, 256), (256, 1), 0), permute_341, out=buf275)
        del permute_341
        buf276 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (256, 512), (1, 256), 0), view_132, out=buf276)
        del view_132
        buf277 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf274, buf277, 1024, 128, grid=grid(1024), stream=stream0)
        buf278 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf277, buf278, 256, 4, grid=grid(256), stream=stream0)
        buf279 = reinterpret_tensor(buf275, (1, 512, 1024), (524288, 1024, 1), 0); del buf275  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf279, addmm_35, 524288, grid=grid(524288), stream=stream0)
        del addmm_35
        buf280 = reinterpret_tensor(buf274, (512, 256), (256, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (512, 1024), (1024, 1), 0), permute_345, out=buf280)
        del permute_345
        buf281 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (1024, 512), (1, 1024), 0), view_130, out=buf281)
        del view_130
        buf282 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf279, buf282, 4096, 128, grid=grid(4096), stream=stream0)
        buf283 = reinterpret_tensor(buf277, (1, 1024), (1024, 1), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf282, buf283, 1024, 4, grid=grid(1024), stream=stream0)
        buf286 = reinterpret_tensor(buf264, (1, 512, 256), (131072, 256, 1), 0); del buf264  # reuse
        buf289 = reinterpret_tensor(buf260, (1, 512, 256), (131072, 256, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf271, buf280, primals_96, mul_38, div_46, getitem_57, buf286, buf289, 512, 256, grid=grid(512), stream=stream0)
        del div_46
        del getitem_57
        del primals_96
        buf287 = empty((256, ), device='cuda', dtype=torch.float32)
        buf288 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf271, buf280, mul_38, buf287, buf288, 256, 512, grid=grid(256), stream=stream0)
        del buf271
        del mul_38
        buf290 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (512, 256), (256, 1), 0), permute_349, out=buf290)
        del permute_349
        buf291 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (256, 512), (1, 256), 0), view_128, out=buf291)
        del view_128
        buf292 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf289, buf292, 1024, 128, grid=grid(1024), stream=stream0)
        buf293 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf292, buf293, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf294 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf290, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_169, getitem_170, getitem_171, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_13
        del clone_default_18
        del clone_default_19
        del clone_default_20
        del getitem_169
        del getitem_170
        del getitem_171
        buf295 = buf294[0]
        buf296 = buf294[1]
        buf297 = buf294[2]
        del buf294
        buf298 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (512, 256), (256, 1), 0), permute_361, out=buf298)
        del permute_361
        buf299 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (256, 512), (1, 256), 0), view_112, out=buf299)
        buf300 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf297, buf300, 1024, 128, grid=grid(1024), stream=stream0)
        buf301 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf300, buf301, 256, 4, grid=grid(256), stream=stream0)
        buf302 = reinterpret_tensor(buf297, (512, 256), (256, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (512, 256), (256, 1), 0), permute_366, out=buf302)
        del permute_366
        buf303 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (256, 512), (1, 256), 0), view_112, out=buf303)
        buf304 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf296, buf304, 1024, 128, grid=grid(1024), stream=stream0)
        buf305 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf304, buf305, 256, 4, grid=grid(256), stream=stream0)
        buf306 = reinterpret_tensor(buf296, (512, 256), (256, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (512, 256), (256, 1), 0), permute_370, out=buf306)
        del permute_370
        buf307 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (256, 512), (1, 256), 0), view_112, out=buf307)
        del view_112
        buf308 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf295, buf308, 1024, 128, grid=grid(1024), stream=stream0)
        buf309 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf308, buf309, 256, 4, grid=grid(256), stream=stream0)
        buf313 = reinterpret_tensor(buf295, (1, 512, 256), (131072, 256, 1), 0); del buf295  # reuse
        buf316 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf286, buf298, buf302, buf306, primals_86, mul_36, div_48, getitem_51, buf313, buf316, 512, 256, grid=grid(512), stream=stream0)
        del div_48
        del getitem_51
        del primals_86
        buf314 = empty((256, ), device='cuda', dtype=torch.float32)
        buf315 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf286, buf298, buf302, buf306, mul_36, buf314, buf315, 256, 512, grid=grid(256), stream=stream0)
        del buf286
        del buf298
        del mul_36
        buf317 = reinterpret_tensor(buf279, (512, 1024), (1024, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (512, 256), (256, 1), 0), permute_374, out=buf317)
        del permute_374
        buf318 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (256, 512), (1, 256), 0), view_110, out=buf318)
        del view_110
        buf319 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf316, buf319, 1024, 128, grid=grid(1024), stream=stream0)
        buf320 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf319, buf320, 256, 4, grid=grid(256), stream=stream0)
        buf321 = reinterpret_tensor(buf317, (1, 512, 1024), (524288, 1024, 1), 0); del buf317  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf321, addmm_29, 524288, grid=grid(524288), stream=stream0)
        del addmm_29
        buf322 = reinterpret_tensor(buf316, (512, 256), (256, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (512, 1024), (1024, 1), 0), permute_378, out=buf322)
        del permute_378
        buf323 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (1024, 512), (1, 1024), 0), view_108, out=buf323)
        del view_108
        buf324 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf321, buf324, 4096, 128, grid=grid(4096), stream=stream0)
        buf325 = reinterpret_tensor(buf319, (1, 1024), (1024, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf324, buf325, 1024, 4, grid=grid(1024), stream=stream0)
        buf328 = reinterpret_tensor(buf306, (1, 512, 256), (131072, 256, 1), 0); del buf306  # reuse
        buf331 = reinterpret_tensor(buf302, (1, 512, 256), (131072, 256, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf313, buf322, primals_80, mul_31, div_49, getitem_47, buf328, buf331, 512, 256, grid=grid(512), stream=stream0)
        del div_49
        del getitem_47
        del primals_80
        buf329 = empty((256, ), device='cuda', dtype=torch.float32)
        buf330 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf313, buf322, mul_31, buf329, buf330, 256, 512, grid=grid(256), stream=stream0)
        del buf313
        del mul_31
        buf332 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (512, 256), (256, 1), 0), permute_382, out=buf332)
        del permute_382
        buf333 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (256, 512), (1, 256), 0), view_106, out=buf333)
        del view_106
        buf334 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf331, buf334, 1024, 128, grid=grid(1024), stream=stream0)
        buf335 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf334, buf335, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf336 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf332, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_176, getitem_177, getitem_178, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_15
        del clone_default_21
        del clone_default_22
        del clone_default_23
        del getitem_176
        del getitem_177
        del getitem_178
        buf337 = buf336[0]
        buf338 = buf336[1]
        buf339 = buf336[2]
        del buf336
        buf340 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (512, 256), (256, 1), 0), permute_394, out=buf340)
        del permute_394
        buf341 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (256, 512), (1, 256), 0), view_90, out=buf341)
        buf342 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf339, buf342, 1024, 128, grid=grid(1024), stream=stream0)
        buf343 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf342, buf343, 256, 4, grid=grid(256), stream=stream0)
        buf344 = reinterpret_tensor(buf339, (512, 256), (256, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (512, 256), (256, 1), 0), permute_399, out=buf344)
        del permute_399
        buf345 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (256, 512), (1, 256), 0), view_90, out=buf345)
        buf346 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf338, buf346, 1024, 128, grid=grid(1024), stream=stream0)
        buf347 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf346, buf347, 256, 4, grid=grid(256), stream=stream0)
        buf348 = reinterpret_tensor(buf338, (512, 256), (256, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (512, 256), (256, 1), 0), permute_403, out=buf348)
        del permute_403
        buf349 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (256, 512), (1, 256), 0), view_90, out=buf349)
        del view_90
        buf350 = buf346; del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf337, buf350, 1024, 128, grid=grid(1024), stream=stream0)
        buf351 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf350, buf351, 256, 4, grid=grid(256), stream=stream0)
        buf355 = reinterpret_tensor(buf337, (1, 512, 256), (131072, 256, 1), 0); del buf337  # reuse
        buf358 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf328, buf340, buf344, buf348, primals_70, mul_29, div_51, getitem_41, buf355, buf358, 512, 256, grid=grid(512), stream=stream0)
        del div_51
        del getitem_41
        del primals_70
        buf356 = empty((256, ), device='cuda', dtype=torch.float32)
        buf357 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf328, buf340, buf344, buf348, mul_29, buf356, buf357, 256, 512, grid=grid(256), stream=stream0)
        del buf328
        del buf340
        del mul_29
        buf359 = reinterpret_tensor(buf321, (512, 1024), (1024, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf358, (512, 256), (256, 1), 0), permute_407, out=buf359)
        del permute_407
        buf360 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf358, (256, 512), (1, 256), 0), view_88, out=buf360)
        del view_88
        buf361 = buf350; del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf358, buf361, 1024, 128, grid=grid(1024), stream=stream0)
        buf362 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf361, buf362, 256, 4, grid=grid(256), stream=stream0)
        buf363 = reinterpret_tensor(buf359, (1, 512, 1024), (524288, 1024, 1), 0); del buf359  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf363, addmm_23, 524288, grid=grid(524288), stream=stream0)
        del addmm_23
        buf364 = reinterpret_tensor(buf358, (512, 256), (256, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (512, 1024), (1024, 1), 0), permute_411, out=buf364)
        del permute_411
        buf365 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (1024, 512), (1, 1024), 0), view_86, out=buf365)
        del view_86
        buf366 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf363, buf366, 4096, 128, grid=grid(4096), stream=stream0)
        buf367 = reinterpret_tensor(buf361, (1, 1024), (1024, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf366, buf367, 1024, 4, grid=grid(1024), stream=stream0)
        buf370 = reinterpret_tensor(buf348, (1, 512, 256), (131072, 256, 1), 0); del buf348  # reuse
        buf373 = reinterpret_tensor(buf344, (1, 512, 256), (131072, 256, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf355, buf364, primals_64, mul_24, div_52, getitem_37, buf370, buf373, 512, 256, grid=grid(512), stream=stream0)
        del div_52
        del getitem_37
        del primals_64
        buf371 = empty((256, ), device='cuda', dtype=torch.float32)
        buf372 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf355, buf364, mul_24, buf371, buf372, 256, 512, grid=grid(256), stream=stream0)
        del buf355
        del mul_24
        buf374 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (512, 256), (256, 1), 0), permute_415, out=buf374)
        del permute_415
        buf375 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (256, 512), (1, 256), 0), view_84, out=buf375)
        del view_84
        buf376 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf373, buf376, 1024, 128, grid=grid(1024), stream=stream0)
        buf377 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf376, buf377, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf378 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf374, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_183, getitem_184, getitem_185, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_17
        del clone_default_24
        del clone_default_25
        del clone_default_26
        del getitem_183
        del getitem_184
        del getitem_185
        buf379 = buf378[0]
        buf380 = buf378[1]
        buf381 = buf378[2]
        del buf378
        buf382 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (512, 256), (256, 1), 0), permute_427, out=buf382)
        del permute_427
        buf383 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (256, 512), (1, 256), 0), view_68, out=buf383)
        buf384 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf381, buf384, 1024, 128, grid=grid(1024), stream=stream0)
        buf385 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf384, buf385, 256, 4, grid=grid(256), stream=stream0)
        buf386 = reinterpret_tensor(buf381, (512, 256), (256, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (512, 256), (256, 1), 0), permute_432, out=buf386)
        del permute_432
        buf387 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (256, 512), (1, 256), 0), view_68, out=buf387)
        buf388 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf380, buf388, 1024, 128, grid=grid(1024), stream=stream0)
        buf389 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf388, buf389, 256, 4, grid=grid(256), stream=stream0)
        buf390 = reinterpret_tensor(buf380, (512, 256), (256, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf379, (512, 256), (256, 1), 0), permute_436, out=buf390)
        del permute_436
        buf391 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf379, (256, 512), (1, 256), 0), view_68, out=buf391)
        del view_68
        buf392 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf379, buf392, 1024, 128, grid=grid(1024), stream=stream0)
        buf393 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf392, buf393, 256, 4, grid=grid(256), stream=stream0)
        buf397 = reinterpret_tensor(buf379, (1, 512, 256), (131072, 256, 1), 0); del buf379  # reuse
        buf400 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf370, buf382, buf386, buf390, primals_54, mul_22, div_54, getitem_31, buf397, buf400, 512, 256, grid=grid(512), stream=stream0)
        del div_54
        del getitem_31
        del primals_54
        buf398 = empty((256, ), device='cuda', dtype=torch.float32)
        buf399 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf370, buf382, buf386, buf390, mul_22, buf398, buf399, 256, 512, grid=grid(256), stream=stream0)
        del buf370
        del buf382
        del mul_22
        buf401 = reinterpret_tensor(buf363, (512, 1024), (1024, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (512, 256), (256, 1), 0), permute_440, out=buf401)
        del permute_440
        buf402 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (256, 512), (1, 256), 0), view_66, out=buf402)
        del view_66
        buf403 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf400, buf403, 1024, 128, grid=grid(1024), stream=stream0)
        buf404 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf403, buf404, 256, 4, grid=grid(256), stream=stream0)
        buf405 = reinterpret_tensor(buf401, (1, 512, 1024), (524288, 1024, 1), 0); del buf401  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf405, addmm_17, 524288, grid=grid(524288), stream=stream0)
        del addmm_17
        buf406 = reinterpret_tensor(buf400, (512, 256), (256, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (512, 1024), (1024, 1), 0), permute_444, out=buf406)
        del permute_444
        buf407 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (1024, 512), (1, 1024), 0), view_64, out=buf407)
        del view_64
        buf408 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf405, buf408, 4096, 128, grid=grid(4096), stream=stream0)
        buf409 = reinterpret_tensor(buf403, (1, 1024), (1024, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf408, buf409, 1024, 4, grid=grid(1024), stream=stream0)
        buf412 = reinterpret_tensor(buf390, (1, 512, 256), (131072, 256, 1), 0); del buf390  # reuse
        buf415 = reinterpret_tensor(buf386, (1, 512, 256), (131072, 256, 1), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf397, buf406, primals_48, mul_17, div_55, getitem_27, buf412, buf415, 512, 256, grid=grid(512), stream=stream0)
        del div_55
        del getitem_27
        del primals_48
        buf413 = empty((256, ), device='cuda', dtype=torch.float32)
        buf414 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf397, buf406, mul_17, buf413, buf414, 256, 512, grid=grid(256), stream=stream0)
        del buf397
        del mul_17
        buf416 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (512, 256), (256, 1), 0), permute_448, out=buf416)
        del permute_448
        buf417 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (256, 512), (1, 256), 0), view_62, out=buf417)
        del view_62
        buf418 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf415, buf418, 1024, 128, grid=grid(1024), stream=stream0)
        buf419 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf418, buf419, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf420 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf416, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_190, getitem_191, getitem_192, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_19
        del clone_default_27
        del clone_default_28
        del clone_default_29
        del getitem_190
        del getitem_191
        del getitem_192
        buf421 = buf420[0]
        buf422 = buf420[1]
        buf423 = buf420[2]
        del buf420
        buf424 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (512, 256), (256, 1), 0), permute_460, out=buf424)
        del permute_460
        buf425 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (256, 512), (1, 256), 0), view_46, out=buf425)
        buf426 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf423, buf426, 1024, 128, grid=grid(1024), stream=stream0)
        buf427 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf426, buf427, 256, 4, grid=grid(256), stream=stream0)
        buf428 = reinterpret_tensor(buf423, (512, 256), (256, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (512, 256), (256, 1), 0), permute_465, out=buf428)
        del permute_465
        buf429 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (256, 512), (1, 256), 0), view_46, out=buf429)
        buf430 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf422, buf430, 1024, 128, grid=grid(1024), stream=stream0)
        buf431 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf430, buf431, 256, 4, grid=grid(256), stream=stream0)
        buf432 = reinterpret_tensor(buf422, (512, 256), (256, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (512, 256), (256, 1), 0), permute_469, out=buf432)
        del permute_469
        buf433 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (256, 512), (1, 256), 0), view_46, out=buf433)
        del view_46
        buf434 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf421, buf434, 1024, 128, grid=grid(1024), stream=stream0)
        buf435 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf434, buf435, 256, 4, grid=grid(256), stream=stream0)
        buf439 = reinterpret_tensor(buf421, (1, 512, 256), (131072, 256, 1), 0); del buf421  # reuse
        buf442 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf412, buf424, buf428, buf432, primals_38, mul_15, div_57, getitem_21, buf439, buf442, 512, 256, grid=grid(512), stream=stream0)
        del div_57
        del getitem_21
        del primals_38
        buf440 = empty((256, ), device='cuda', dtype=torch.float32)
        buf441 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf412, buf424, buf428, buf432, mul_15, buf440, buf441, 256, 512, grid=grid(256), stream=stream0)
        del buf412
        del buf424
        del mul_15
        buf443 = reinterpret_tensor(buf405, (512, 1024), (1024, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf442, (512, 256), (256, 1), 0), permute_473, out=buf443)
        del permute_473
        buf444 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf442, (256, 512), (1, 256), 0), view_44, out=buf444)
        del view_44
        buf445 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf442, buf445, 1024, 128, grid=grid(1024), stream=stream0)
        buf446 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf445, buf446, 256, 4, grid=grid(256), stream=stream0)
        buf447 = reinterpret_tensor(buf443, (1, 512, 1024), (524288, 1024, 1), 0); del buf443  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf447, addmm_11, 524288, grid=grid(524288), stream=stream0)
        del addmm_11
        buf448 = reinterpret_tensor(buf442, (512, 256), (256, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (512, 1024), (1024, 1), 0), permute_477, out=buf448)
        del permute_477
        buf449 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (1024, 512), (1, 1024), 0), view_42, out=buf449)
        del view_42
        buf450 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf447, buf450, 4096, 128, grid=grid(4096), stream=stream0)
        buf451 = reinterpret_tensor(buf445, (1, 1024), (1024, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf450, buf451, 1024, 4, grid=grid(1024), stream=stream0)
        buf454 = reinterpret_tensor(buf432, (1, 512, 256), (131072, 256, 1), 0); del buf432  # reuse
        buf457 = reinterpret_tensor(buf428, (1, 512, 256), (131072, 256, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf439, buf448, primals_32, mul_10, div_58, getitem_17, buf454, buf457, 512, 256, grid=grid(512), stream=stream0)
        del div_58
        del getitem_17
        del primals_32
        buf455 = empty((256, ), device='cuda', dtype=torch.float32)
        buf456 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf439, buf448, mul_10, buf455, buf456, 256, 512, grid=grid(256), stream=stream0)
        del buf439
        del mul_10
        buf458 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (512, 256), (256, 1), 0), permute_481, out=buf458)
        del permute_481
        buf459 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (256, 512), (1, 256), 0), view_40, out=buf459)
        del view_40
        buf460 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf457, buf460, 1024, 128, grid=grid(1024), stream=stream0)
        buf461 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf460, buf461, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf462 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf458, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_197, getitem_198, getitem_199, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_21
        del clone_default_30
        del clone_default_31
        del clone_default_32
        del getitem_197
        del getitem_198
        del getitem_199
        buf463 = buf462[0]
        buf464 = buf462[1]
        buf465 = buf462[2]
        del buf462
        buf466 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (512, 256), (256, 1), 0), permute_493, out=buf466)
        del permute_493
        buf467 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (256, 512), (1, 256), 0), view_24, out=buf467)
        buf468 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf465, buf468, 1024, 128, grid=grid(1024), stream=stream0)
        buf469 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf468, buf469, 256, 4, grid=grid(256), stream=stream0)
        buf470 = reinterpret_tensor(buf465, (512, 256), (256, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (512, 256), (256, 1), 0), permute_498, out=buf470)
        del permute_498
        buf471 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (256, 512), (1, 256), 0), view_24, out=buf471)
        buf472 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf464, buf472, 1024, 128, grid=grid(1024), stream=stream0)
        buf473 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf472, buf473, 256, 4, grid=grid(256), stream=stream0)
        buf474 = reinterpret_tensor(buf464, (512, 256), (256, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (512, 256), (256, 1), 0), permute_502, out=buf474)
        del permute_502
        buf475 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (256, 512), (1, 256), 0), view_24, out=buf475)
        del view_24
        buf476 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf463, buf476, 1024, 128, grid=grid(1024), stream=stream0)
        buf477 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf476, buf477, 256, 4, grid=grid(256), stream=stream0)
        buf481 = reinterpret_tensor(buf463, (1, 512, 256), (131072, 256, 1), 0); del buf463  # reuse
        buf484 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_17.run(buf454, buf466, buf470, buf474, primals_22, mul_8, div_60, getitem_11, buf481, buf484, 512, 256, grid=grid(512), stream=stream0)
        del div_60
        del getitem_11
        del primals_22
        buf482 = empty((256, ), device='cuda', dtype=torch.float32)
        buf483 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf454, buf466, buf470, buf474, mul_8, buf482, buf483, 256, 512, grid=grid(256), stream=stream0)
        del buf454
        del buf466
        del mul_8
        buf485 = reinterpret_tensor(buf447, (512, 1024), (1024, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (512, 256), (256, 1), 0), permute_506, out=buf485)
        del permute_506
        buf486 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (256, 512), (1, 256), 0), view_22, out=buf486)
        del view_22
        buf487 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf484, buf487, 1024, 128, grid=grid(1024), stream=stream0)
        buf488 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf487, buf488, 256, 4, grid=grid(256), stream=stream0)
        buf489 = reinterpret_tensor(buf485, (1, 512, 1024), (524288, 1024, 1), 0); del buf485  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_12.run(buf489, addmm_5, 524288, grid=grid(524288), stream=stream0)
        del addmm_5
        buf490 = reinterpret_tensor(buf484, (512, 256), (256, 1), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf489, (512, 1024), (1024, 1), 0), permute_510, out=buf490)
        del permute_510
        buf491 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf489, (1024, 512), (1, 1024), 0), view_20, out=buf491)
        del view_20
        buf492 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf489, buf492, 4096, 128, grid=grid(4096), stream=stream0)
        del buf489
        buf493 = reinterpret_tensor(buf487, (1, 1024), (1024, 1), 0); del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf492, buf493, 1024, 4, grid=grid(1024), stream=stream0)
        del buf492
        buf496 = reinterpret_tensor(buf474, (1, 512, 256), (131072, 256, 1), 0); del buf474  # reuse
        buf499 = reinterpret_tensor(buf470, (1, 512, 256), (131072, 256, 1), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_15.run(buf481, buf490, primals_16, mul_3, div_61, getitem_7, buf496, buf499, 512, 256, grid=grid(512), stream=stream0)
        del div_61
        del getitem_7
        del primals_16
        buf497 = empty((256, ), device='cuda', dtype=torch.float32)
        buf498 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf481, buf490, mul_3, buf497, buf498, 256, 512, grid=grid(256), stream=stream0)
        del buf481
        del mul_3
        buf500 = buf490; del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (512, 256), (256, 1), 0), permute_514, out=buf500)
        del permute_514
        buf501 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (256, 512), (1, 256), 0), view_18, out=buf501)
        del view_18
        buf502 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf499, buf502, 1024, 128, grid=grid(1024), stream=stream0)
        del buf499
        buf503 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf502, buf503, 256, 4, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf504 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf500, (1, 4, 512, 64), (131072, 64, 256, 1), 0), clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_204, getitem_205, getitem_206, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_23
        del clone_default_33
        del clone_default_34
        del clone_default_35
        del getitem_204
        del getitem_205
        del getitem_206
        buf505 = buf504[0]
        buf506 = buf504[1]
        buf507 = buf504[2]
        del buf504
        buf508 = buf500; del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf507, (512, 256), (256, 1), 0), permute_526, out=buf508)
        del permute_526
        buf509 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf507, (256, 512), (1, 256), 0), view_2, out=buf509)
        buf510 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf507, buf510, 1024, 128, grid=grid(1024), stream=stream0)
        buf511 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf510, buf511, 256, 4, grid=grid(256), stream=stream0)
        buf512 = reinterpret_tensor(buf507, (512, 256), (256, 1), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (512, 256), (256, 1), 0), permute_531, out=buf512)
        del permute_531
        buf513 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (256, 512), (1, 256), 0), view_2, out=buf513)
        buf514 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf506, buf514, 1024, 128, grid=grid(1024), stream=stream0)
        buf515 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf514, buf515, 256, 4, grid=grid(256), stream=stream0)
        buf516 = reinterpret_tensor(buf506, (512, 256), (256, 1), 0); del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf505, (512, 256), (256, 1), 0), permute_535, out=buf516)
        del permute_535
        buf517 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf505, (256, 512), (1, 256), 0), view_2, out=buf517)
        del view_2
        buf518 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf505, buf518, 1024, 128, grid=grid(1024), stream=stream0)
        del buf505
        buf519 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf518, buf519, 256, 4, grid=grid(256), stream=stream0)
        buf520 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf520, buf508, buf512, buf516, 131072, grid=grid(131072), stream=stream0)
        del buf508
        del buf512
        del buf516
        buf521 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (512, 256), (256, 1), 0), permute_539, out=buf521)
        del permute_539
        buf522 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (256, 512), (1, 256), 0), view, out=buf522)
        del view
        buf523 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf520, buf523, 1024, 128, grid=grid(1024), stream=stream0)
        del buf520
        buf524 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf523, buf524, 256, 4, grid=grid(256), stream=stream0)
        del buf523
        buf531 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf535 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        buf539 = empty((1, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_20.run(buf521, getitem_3, primals_4, mul_1, div_63, slice_4, expand, primals_209, buf531, buf535, buf539, 512, 128, grid=grid(512), stream=stream0)
        del div_63
        del primals_4
        buf528 = empty((128, ), device='cuda', dtype=torch.float32)
        buf529 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_21.run(buf521, getitem_3, mul_1, buf528, buf529, 128, 512, grid=grid(128), stream=stream0)
        del getitem_3
        del mul_1
        buf530 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_22.run(buf530, 65536, grid=grid(65536), stream=stream0)
        aten.index_put_(buf530, [slice_4], buf531, True)
        del buf531
        del slice_4
        buf534 = empty((2, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_23.run(buf534, 256, grid=grid(256), stream=stream0)
        aten.index_put_(buf534, [expand], buf535, True)
        del buf535
        del expand
        buf538 = empty((30522, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_24.run(buf538, 3906816, grid=grid(3906816), stream=stream0)
        aten.index_put_(buf538, [primals_209], buf539, True)
        del buf539
        del primals_209
        return (buf538, buf534, buf530, buf528, buf529, reinterpret_tensor(buf522, (256, 128), (128, 1), 0), reinterpret_tensor(buf524, (256, ), (1, ), 0), reinterpret_tensor(buf517, (256, 256), (256, 1), 0), reinterpret_tensor(buf519, (256, ), (1, ), 0), reinterpret_tensor(buf513, (256, 256), (256, 1), 0), reinterpret_tensor(buf515, (256, ), (1, ), 0), reinterpret_tensor(buf509, (256, 256), (256, 1), 0), reinterpret_tensor(buf511, (256, ), (1, ), 0), reinterpret_tensor(buf501, (256, 256), (256, 1), 0), reinterpret_tensor(buf503, (256, ), (1, ), 0), buf497, buf498, reinterpret_tensor(buf491, (1024, 256), (256, 1), 0), reinterpret_tensor(buf493, (1024, ), (1, ), 0), reinterpret_tensor(buf486, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf488, (256, ), (1, ), 0), buf482, buf483, reinterpret_tensor(buf475, (256, 256), (256, 1), 0), reinterpret_tensor(buf477, (256, ), (1, ), 0), reinterpret_tensor(buf471, (256, 256), (256, 1), 0), reinterpret_tensor(buf473, (256, ), (1, ), 0), reinterpret_tensor(buf467, (256, 256), (256, 1), 0), reinterpret_tensor(buf469, (256, ), (1, ), 0), reinterpret_tensor(buf459, (256, 256), (256, 1), 0), reinterpret_tensor(buf461, (256, ), (1, ), 0), buf455, buf456, reinterpret_tensor(buf449, (1024, 256), (256, 1), 0), reinterpret_tensor(buf451, (1024, ), (1, ), 0), reinterpret_tensor(buf444, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf446, (256, ), (1, ), 0), buf440, buf441, reinterpret_tensor(buf433, (256, 256), (256, 1), 0), reinterpret_tensor(buf435, (256, ), (1, ), 0), reinterpret_tensor(buf429, (256, 256), (256, 1), 0), reinterpret_tensor(buf431, (256, ), (1, ), 0), reinterpret_tensor(buf425, (256, 256), (256, 1), 0), reinterpret_tensor(buf427, (256, ), (1, ), 0), reinterpret_tensor(buf417, (256, 256), (256, 1), 0), reinterpret_tensor(buf419, (256, ), (1, ), 0), buf413, buf414, reinterpret_tensor(buf407, (1024, 256), (256, 1), 0), reinterpret_tensor(buf409, (1024, ), (1, ), 0), reinterpret_tensor(buf402, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf404, (256, ), (1, ), 0), buf398, buf399, reinterpret_tensor(buf391, (256, 256), (256, 1), 0), reinterpret_tensor(buf393, (256, ), (1, ), 0), reinterpret_tensor(buf387, (256, 256), (256, 1), 0), reinterpret_tensor(buf389, (256, ), (1, ), 0), reinterpret_tensor(buf383, (256, 256), (256, 1), 0), reinterpret_tensor(buf385, (256, ), (1, ), 0), reinterpret_tensor(buf375, (256, 256), (256, 1), 0), reinterpret_tensor(buf377, (256, ), (1, ), 0), buf371, buf372, reinterpret_tensor(buf365, (1024, 256), (256, 1), 0), reinterpret_tensor(buf367, (1024, ), (1, ), 0), reinterpret_tensor(buf360, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf362, (256, ), (1, ), 0), buf356, buf357, reinterpret_tensor(buf349, (256, 256), (256, 1), 0), reinterpret_tensor(buf351, (256, ), (1, ), 0), reinterpret_tensor(buf345, (256, 256), (256, 1), 0), reinterpret_tensor(buf347, (256, ), (1, ), 0), reinterpret_tensor(buf341, (256, 256), (256, 1), 0), reinterpret_tensor(buf343, (256, ), (1, ), 0), reinterpret_tensor(buf333, (256, 256), (256, 1), 0), reinterpret_tensor(buf335, (256, ), (1, ), 0), buf329, buf330, reinterpret_tensor(buf323, (1024, 256), (256, 1), 0), reinterpret_tensor(buf325, (1024, ), (1, ), 0), reinterpret_tensor(buf318, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf320, (256, ), (1, ), 0), buf314, buf315, reinterpret_tensor(buf307, (256, 256), (256, 1), 0), reinterpret_tensor(buf309, (256, ), (1, ), 0), reinterpret_tensor(buf303, (256, 256), (256, 1), 0), reinterpret_tensor(buf305, (256, ), (1, ), 0), reinterpret_tensor(buf299, (256, 256), (256, 1), 0), reinterpret_tensor(buf301, (256, ), (1, ), 0), reinterpret_tensor(buf291, (256, 256), (256, 1), 0), reinterpret_tensor(buf293, (256, ), (1, ), 0), buf287, buf288, reinterpret_tensor(buf281, (1024, 256), (256, 1), 0), reinterpret_tensor(buf283, (1024, ), (1, ), 0), reinterpret_tensor(buf276, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf278, (256, ), (1, ), 0), buf272, buf273, reinterpret_tensor(buf265, (256, 256), (256, 1), 0), reinterpret_tensor(buf267, (256, ), (1, ), 0), reinterpret_tensor(buf261, (256, 256), (256, 1), 0), reinterpret_tensor(buf263, (256, ), (1, ), 0), reinterpret_tensor(buf257, (256, 256), (256, 1), 0), reinterpret_tensor(buf259, (256, ), (1, ), 0), reinterpret_tensor(buf249, (256, 256), (256, 1), 0), reinterpret_tensor(buf251, (256, ), (1, ), 0), buf245, buf246, reinterpret_tensor(buf239, (1024, 256), (256, 1), 0), reinterpret_tensor(buf241, (1024, ), (1, ), 0), reinterpret_tensor(buf234, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf236, (256, ), (1, ), 0), buf230, buf231, reinterpret_tensor(buf223, (256, 256), (256, 1), 0), reinterpret_tensor(buf225, (256, ), (1, ), 0), reinterpret_tensor(buf219, (256, 256), (256, 1), 0), reinterpret_tensor(buf221, (256, ), (1, ), 0), reinterpret_tensor(buf215, (256, 256), (256, 1), 0), reinterpret_tensor(buf217, (256, ), (1, ), 0), reinterpret_tensor(buf207, (256, 256), (256, 1), 0), reinterpret_tensor(buf209, (256, ), (1, ), 0), buf203, buf204, reinterpret_tensor(buf197, (1024, 256), (256, 1), 0), reinterpret_tensor(buf199, (1024, ), (1, ), 0), reinterpret_tensor(buf192, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf194, (256, ), (1, ), 0), buf188, buf189, reinterpret_tensor(buf181, (256, 256), (256, 1), 0), reinterpret_tensor(buf183, (256, ), (1, ), 0), reinterpret_tensor(buf177, (256, 256), (256, 1), 0), reinterpret_tensor(buf179, (256, ), (1, ), 0), reinterpret_tensor(buf173, (256, 256), (256, 1), 0), reinterpret_tensor(buf175, (256, ), (1, ), 0), reinterpret_tensor(buf165, (256, 256), (256, 1), 0), reinterpret_tensor(buf167, (256, ), (1, ), 0), buf161, buf162, reinterpret_tensor(buf155, (1024, 256), (256, 1), 0), reinterpret_tensor(buf157, (1024, ), (1, ), 0), reinterpret_tensor(buf150, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf152, (256, ), (1, ), 0), buf146, buf147, reinterpret_tensor(buf139, (256, 256), (256, 1), 0), reinterpret_tensor(buf141, (256, ), (1, ), 0), reinterpret_tensor(buf135, (256, 256), (256, 1), 0), reinterpret_tensor(buf137, (256, ), (1, ), 0), reinterpret_tensor(buf131, (256, 256), (256, 1), 0), reinterpret_tensor(buf133, (256, ), (1, ), 0), reinterpret_tensor(buf123, (256, 256), (256, 1), 0), reinterpret_tensor(buf125, (256, ), (1, ), 0), buf119, buf120, reinterpret_tensor(buf113, (1024, 256), (256, 1), 0), reinterpret_tensor(buf115, (1024, ), (1, ), 0), reinterpret_tensor(buf108, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf110, (256, ), (1, ), 0), buf104, buf105, reinterpret_tensor(buf97, (256, 256), (256, 1), 0), reinterpret_tensor(buf99, (256, ), (1, ), 0), reinterpret_tensor(buf93, (256, 256), (256, 1), 0), reinterpret_tensor(buf95, (256, ), (1, ), 0), reinterpret_tensor(buf89, (256, 256), (256, 1), 0), reinterpret_tensor(buf91, (256, ), (1, ), 0), reinterpret_tensor(buf81, (256, 256), (256, 1), 0), reinterpret_tensor(buf83, (256, ), (1, ), 0), buf77, buf78, reinterpret_tensor(buf71, (1024, 256), (256, 1), 0), reinterpret_tensor(buf73, (1024, ), (1, ), 0), reinterpret_tensor(buf66, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf68, (256, ), (1, ), 0), buf62, buf63, reinterpret_tensor(buf55, (256, 256), (256, 1), 0), reinterpret_tensor(buf57, (256, ), (1, ), 0), reinterpret_tensor(buf51, (256, 256), (256, 1), 0), reinterpret_tensor(buf53, (256, ), (1, ), 0), reinterpret_tensor(buf47, (256, 256), (256, 1), 0), reinterpret_tensor(buf49, (256, ), (1, ), 0), reinterpret_tensor(buf39, (256, 256), (256, 1), 0), reinterpret_tensor(buf41, (256, ), (1, ), 0), buf35, buf36, reinterpret_tensor(buf29, (1024, 256), (256, 1), 0), reinterpret_tensor(buf31, (1024, ), (1, ), 0), reinterpret_tensor(buf24, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf26, (256, ), (1, ), 0), buf20, buf21, reinterpret_tensor(buf14, (128, 256), (256, 1), 0), reinterpret_tensor(buf16, (128, ), (1, ), 0), buf10, buf11, reinterpret_tensor(buf6, (30522, 128), (128, 1), 0), reinterpret_tensor(buf7, (30522, ), (1, ), 0), None, None, None, None, )


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
    primals_202 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
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
    addmm_73 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_90 = rand_strided((1, 512, 128), (65536, 128, 1), device='cuda:0', dtype=torch.float32)
    view_268 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    sub_40 = rand_strided((511, 30522), (30522, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    ne_3 = rand_strided((511, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_2 = rand_strided((511, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_135 = rand_strided((30522, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_139 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_143 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_147 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_168 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_172 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_176 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_180 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_196 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_201 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_205 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_213 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_217 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_229 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_242 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_246 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_262 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_271 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_275 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_279 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_300 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_304 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_308 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_312 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_316 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_328 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_333 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_337 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_341 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_345 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_349 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_361 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_366 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_370 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_374 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_378 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_382 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_394 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_399 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_403 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_407 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_411 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_427 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_432 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_436 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_440 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_444 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_465 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_469 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_473 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_477 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_481 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_493 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_498 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_502 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_506 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_510 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_514 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_526 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_531 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_535 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_539 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 30522), (15627264, 30522, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, primals_112, primals_118, primals_128, primals_134, primals_144, primals_150, primals_160, primals_166, primals_176, primals_182, primals_192, primals_198, primals_202, primals_209, expand, slice_4, mul_1, getitem_3, view, view_2, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_18, getitem_7, mul_3, view_20, addmm_5, view_22, getitem_11, mul_8, view_24, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_40, getitem_17, mul_10, view_42, addmm_11, view_44, getitem_21, mul_15, view_46, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_62, getitem_27, mul_17, view_64, addmm_17, view_66, getitem_31, mul_22, view_68, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_84, getitem_37, mul_24, view_86, addmm_23, view_88, getitem_41, mul_29, view_90, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_106, getitem_47, mul_31, view_108, addmm_29, view_110, getitem_51, mul_36, view_112, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_128, getitem_57, mul_38, view_130, addmm_35, view_132, getitem_61, mul_43, view_134, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_150, getitem_67, mul_45, view_152, addmm_41, view_154, getitem_71, mul_50, view_156, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_172, getitem_77, mul_52, view_174, addmm_47, view_176, getitem_81, mul_57, view_178, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_194, getitem_87, mul_59, view_196, addmm_53, view_198, getitem_91, mul_64, view_200, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_216, getitem_97, mul_66, view_218, addmm_59, view_220, getitem_101, mul_71, view_222, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_238, getitem_107, mul_73, view_240, addmm_65, view_242, getitem_111, mul_78, view_244, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_260, getitem_117, mul_80, view_262, addmm_71, view_264, getitem_121, mul_85, view_266, addmm_73, mul_90, view_268, sub_40, convert_element_type, ne_3, where_2, permute_135, div_26, permute_139, div_27, permute_143, permute_147, div_28, permute_151, permute_163, permute_168, permute_172, div_30, permute_176, permute_180, div_31, permute_184, permute_196, permute_201, permute_205, div_33, permute_209, permute_213, div_34, permute_217, permute_229, permute_234, permute_238, div_36, permute_242, permute_246, div_37, permute_250, permute_262, permute_267, permute_271, div_39, permute_275, permute_279, div_40, permute_283, permute_295, permute_300, permute_304, div_42, permute_308, permute_312, div_43, permute_316, permute_328, permute_333, permute_337, div_45, permute_341, permute_345, div_46, permute_349, permute_361, permute_366, permute_370, div_48, permute_374, permute_378, div_49, permute_382, permute_394, permute_399, permute_403, div_51, permute_407, permute_411, div_52, permute_415, permute_427, permute_432, permute_436, div_54, permute_440, permute_444, div_55, permute_448, permute_460, permute_465, permute_469, div_57, permute_473, permute_477, div_58, permute_481, permute_493, permute_498, permute_502, div_60, permute_506, permute_510, div_61, permute_514, permute_526, permute_531, permute_535, permute_539, div_63, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ElectraForCausalLM', benchmark_compiled_module)
