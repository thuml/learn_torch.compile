
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


# kernel path: /tmp/torchinductor_youkaichao/te/cte4qfy235pcdii6wiimwer3ez7tkrbbed3l3uqy5znqoghmcmk6.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_12
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25681327
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


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2gszwo642ihfcjw4q2ye7iyu4i2av23dbwwxqybe76qwscexqj.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_12
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 511
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + x0), xmask)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzdhszkyzvtp2dfdfxock4ja2ofkrqaf5kdxre4hg5bwbbkxxer.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_13
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 511
    rnumel = 50257
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (1 + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50257*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctq7epkewfp2itendpqyrcmtegr7gzfp2rlsl3pyjnivts6o7nrs.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25731584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 50257)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp8 = tl.load(in_ptr3 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp1 = x1
    tmp2 = tl.full([1], 511, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x2), tmp3 & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (1 + x1), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full([1], -100, tl.int64)
    tmp7 = tmp5 != tmp6
    tmp12 = tmp9 / tmp11
    tmp13 = 0.0
    tmp14 = tl.where(tmp7, tmp12, tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tl.load(in_ptr5 + (x2), tmp3 & xmask, other=0.0)
    tmp17 = tl.exp(tmp16)
    tmp18 = tl.load(in_ptr6 + (x1), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp15 - tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp3, tmp20, tmp21)
    tmp23 = tl.where(tmp3, tmp22, tmp13)
    tmp24 = tmp0 + tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ln/cln4llzqt5vz3rzpgtndckesulclxlxzqml3iroicr23icl3ahir.py
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 768.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp19 * tmp23
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cuj5jipddg65hcmq25dutvuyerqmzocp5m7uy3uhaqqpiq5zmj6v.py
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5e32ob646uag7xx6bcm77p3cbdhfinmau2k2fzb6vsgphniet6.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_6', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnu4hetcmyv2laomty752vknskibjjygllxzp5ciznvl4mugsv5.py
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
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ed/ced5lc7p6i5j7rz2opy4nabtsqd45jkf6jwphx5d45segvij7whn.py
# Source Nodes: [add_23, mul_20], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
# add_23 => add_47
# mul_20 => mul_44
triton_poi_fused_add_mul_pow_tanh_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/gl/cgl6bayeu5igh4yr3tnjcvvir3mdmezr7idjqnxuhbu43giwmcbi.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2o/c2onzperwjniaakuelmjplflpm45gjgsaqljxv745zepe6jcs2jw.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_10', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxe2f45xdiwnk36d2kkohmjis3ueq5ijcizg7xz6brzlizgizv4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 768.0
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
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cugqsixtmsb2p5jxkybvv6amdwtvwr6ztbqljifc6zqeo22uez4x.py
# Source Nodes: [full, loss], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.nll_loss_forward, aten.where]
# full => full_default
# loss => full_default_13
triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
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
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (1024*x2)), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp6 * tmp11
    tmp14 = tmp7 - tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp18, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z5/cz5ow7ykljkej2eerbs5orcu5kyi5gqwjuss67wxb4tqsre54wgp.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = x1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*y0) + (32768*((x1 // 64) % 12)) + (x1 % 64)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 1536, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((64*y0) + (32768*((x1 // 64) % 12)) + (x1 % 64)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr2 + (y0 + (512*(x1 % 768))), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tmp0 >= tmp9
    tmp18 = tl.full([1, 1], 2304, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr3 + ((64*y0) + (32768*((x1 // 64) % 12)) + (x1 % 64)), tmp17 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr4 + ((64*y0) + (32768*((x1 // 64) % 12)) + (x1 % 64)), tmp17 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp17, tmp22, tmp23)
    tmp25 = tl.where(tmp11, tmp16, tmp24)
    tmp26 = tl.where(tmp4, tmp7, tmp25)
    tl.store(out_ptr0 + (x1 + (2304*y0)), tmp26, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c253imu6sszywwfmdblnsbavonw5ynyzfksyucd6gf3dm53xeaev.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2304
    x1 = (xindex // 2304)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2304*r2) + (294912*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjjus3ur4lrynaypkujxprmp7nardyob3jukfabzjzcsnsq6b3y.py
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
    size_hints=[4096, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2304*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ep/cep4in4ecc5q6g2yryuim2me5imegjr2qgjhal6cijemfgjdaf55.py
# Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# loss => full_default_13
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp30 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 768.0
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
    tmp27 = tl.full([1], False, tl.int1)
    tmp28 = 0.0
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp31 = tl.full([1], -1, tl.int64)
    tmp32 = tmp30 == tmp31
    tmp33 = tl.where(tmp32, tmp28, tmp26)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp33, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ce/cceinhfapo7yes6sglu6amig7j4ytfc6ndzqmey4s5ojuhcxph3k.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgouzyuggo3zy3umtyihzvstnpzfbzg64fbeqon3dldjsfapn4t.py
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
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38597376
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
    primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_85, view, view_1, getitem_1, mul, slice_4, getitem_8, getitem_10, mul_2, addmm_2, tanh, getitem_14, mul_8, slice_8, getitem_21, getitem_23, mul_10, addmm_6, tanh_1, getitem_27, mul_16, slice_12, getitem_34, getitem_36, mul_18, addmm_10, tanh_2, getitem_40, mul_24, slice_16, getitem_47, getitem_49, mul_26, addmm_14, tanh_3, getitem_53, mul_32, slice_20, getitem_60, getitem_62, mul_34, addmm_18, tanh_4, getitem_66, mul_40, slice_24, getitem_73, getitem_75, mul_42, addmm_22, tanh_5, getitem_79, mul_48, view_111, sub_20, convert_element_type_6, permute_33, div_14, permute_35, permute_36, permute_37, permute_38, div_15, permute_39, permute_40, permute_42, permute_43, alias_15, permute_44, permute_45, permute_50, permute_51, div_17, permute_52, permute_53, permute_54, permute_55, div_18, permute_56, permute_57, permute_59, permute_60, alias_17, permute_61, permute_62, permute_67, permute_68, div_20, permute_69, permute_70, permute_71, permute_72, div_21, permute_73, permute_74, permute_76, permute_77, alias_19, permute_78, permute_79, permute_84, permute_85, div_23, permute_86, permute_87, permute_88, permute_89, div_24, permute_90, permute_91, permute_93, permute_94, alias_21, permute_95, permute_96, permute_101, permute_102, div_26, permute_103, permute_104, permute_105, permute_106, div_27, permute_107, permute_108, permute_110, permute_111, alias_23, permute_112, permute_113, permute_118, permute_119, div_29, permute_120, permute_121, permute_122, permute_123, div_30, permute_124, permute_125, permute_127, permute_128, alias_25, permute_129, permute_130, permute_135, permute_136, div_32, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14 = args
    args.clear()
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_85, (1, 512), (512, 1))
    assert_size_stride(view, (1, 512), (512, 1))
    assert_size_stride(view_1, (1, 512), (512, 1))
    assert_size_stride(getitem_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_4, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_8, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_10, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_2, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_2, (512, 3072), (3072, 1))
    assert_size_stride(tanh, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_14, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_8, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_21, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_23, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_10, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_6, (512, 3072), (3072, 1))
    assert_size_stride(tanh_1, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_16, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_12, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_34, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_18, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(tanh_2, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_40, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_16, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_47, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_49, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_26, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_14, (512, 3072), (3072, 1))
    assert_size_stride(tanh_3, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_53, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_32, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_20, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_60, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_62, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_34, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_18, (512, 3072), (3072, 1))
    assert_size_stride(tanh_4, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_66, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_40, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_24, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_73, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_75, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_42, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(tanh_5, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_79, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_48, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_111, (512, 768), (768, 1))
    assert_size_stride(sub_20, (511, 50257), (50257, 1))
    assert_size_stride(convert_element_type_6, (), ())
    assert_size_stride(permute_33, (50257, 768), (768, 1))
    assert_size_stride(div_14, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_35, (768, 3072), (1, 768))
    assert_size_stride(permute_36, (3072, 512), (1, 3072))
    assert_size_stride(permute_37, (3072, 768), (1, 3072))
    assert_size_stride(permute_38, (768, 512), (1, 768))
    assert_size_stride(div_15, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_39, (768, 768), (1, 768))
    assert_size_stride(permute_40, (768, 512), (1, 768))
    assert_size_stride(permute_42, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_43, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_15, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_44, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_45, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_50, (2304, 768), (1, 2304))
    assert_size_stride(permute_51, (768, 512), (1, 768))
    assert_size_stride(div_17, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_52, (768, 3072), (1, 768))
    assert_size_stride(permute_53, (3072, 512), (1, 3072))
    assert_size_stride(permute_54, (3072, 768), (1, 3072))
    assert_size_stride(permute_55, (768, 512), (1, 768))
    assert_size_stride(div_18, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_56, (768, 768), (1, 768))
    assert_size_stride(permute_57, (768, 512), (1, 768))
    assert_size_stride(permute_59, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_60, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_17, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_61, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_62, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_67, (2304, 768), (1, 2304))
    assert_size_stride(permute_68, (768, 512), (1, 768))
    assert_size_stride(div_20, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_69, (768, 3072), (1, 768))
    assert_size_stride(permute_70, (3072, 512), (1, 3072))
    assert_size_stride(permute_71, (3072, 768), (1, 3072))
    assert_size_stride(permute_72, (768, 512), (1, 768))
    assert_size_stride(div_21, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_73, (768, 768), (1, 768))
    assert_size_stride(permute_74, (768, 512), (1, 768))
    assert_size_stride(permute_76, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_77, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_19, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_78, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_79, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_84, (2304, 768), (1, 2304))
    assert_size_stride(permute_85, (768, 512), (1, 768))
    assert_size_stride(div_23, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_86, (768, 3072), (1, 768))
    assert_size_stride(permute_87, (3072, 512), (1, 3072))
    assert_size_stride(permute_88, (3072, 768), (1, 3072))
    assert_size_stride(permute_89, (768, 512), (1, 768))
    assert_size_stride(div_24, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_90, (768, 768), (1, 768))
    assert_size_stride(permute_91, (768, 512), (1, 768))
    assert_size_stride(permute_93, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_94, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_21, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_95, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_96, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_101, (2304, 768), (1, 2304))
    assert_size_stride(permute_102, (768, 512), (1, 768))
    assert_size_stride(div_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_103, (768, 3072), (1, 768))
    assert_size_stride(permute_104, (3072, 512), (1, 3072))
    assert_size_stride(permute_105, (3072, 768), (1, 3072))
    assert_size_stride(permute_106, (768, 512), (1, 768))
    assert_size_stride(div_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_107, (768, 768), (1, 768))
    assert_size_stride(permute_108, (768, 512), (1, 768))
    assert_size_stride(permute_110, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_111, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_23, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_112, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_113, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_118, (2304, 768), (1, 2304))
    assert_size_stride(permute_119, (768, 512), (1, 768))
    assert_size_stride(div_29, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_120, (768, 3072), (1, 768))
    assert_size_stride(permute_121, (3072, 512), (1, 3072))
    assert_size_stride(permute_122, (3072, 768), (1, 3072))
    assert_size_stride(permute_123, (768, 512), (1, 768))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_124, (768, 768), (1, 768))
    assert_size_stride(permute_125, (768, 512), (1, 768))
    assert_size_stride(permute_127, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_128, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_25, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_129, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_130, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_135, (2304, 768), (1, 2304))
    assert_size_stride(permute_136, (768, 512), (1, 768))
    assert_size_stride(div_32, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 50257), (25731584, 50257, 1))
    assert_size_stride(tangents_3, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_4, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_5, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_6, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_7, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_8, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_9, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_10, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_11, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_12, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_13, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_14, (1, 12, 512, 64), (393216, 32768, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((511, 50257), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 25681327, grid=grid(25681327), stream=stream0)
        buf1 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_85, buf1, 511, grid=grid(511), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf4 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_85, tangents_1, convert_element_type_6, buf4, 511, 50257, grid=grid(511), stream=stream0)
        buf5 = empty((1, 512, 50257), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
        triton_poi_fused_add_slice_backward_3.run(tangents_2, buf0, primals_85, tangents_1, convert_element_type_6, sub_20, buf4, buf5, 25731584, grid=grid(25731584), stream=stream0)
        del buf0
        del buf4
        del convert_element_type_6
        del primals_85
        del sub_20
        del tangents_1
        del tangents_2
        buf6 = empty((50257, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (50257, 512), (1, 50257), 0), view_111, out=buf6)
        del view_111
        buf7 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 50257), (50257, 1), 0), permute_33, out=buf7)
        del buf5
        del permute_33
        buf10 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf13 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_4.run(buf7, primals_75, mul_48, div_14, getitem_79, buf10, buf13, 512, 768, grid=grid(512), stream=stream0)
        del div_14
        del getitem_79
        del primals_75
        buf11 = empty((768, ), device='cuda', dtype=torch.float32)
        buf12 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf7, mul_48, buf11, buf12, 768, 512, grid=grid(768), stream=stream0)
        del mul_48
        buf14 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (512, 768), (768, 1), 0), permute_35, out=buf14)
        del permute_35
        buf15 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_36, reinterpret_tensor(buf13, (512, 768), (768, 1), 0), out=buf15)
        del permute_36
        buf16 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf13, buf16, 3072, 128, grid=grid(3072), stream=stream0)
        buf17 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf16, buf17, 768, 4, grid=grid(768), stream=stream0)
        buf18 = reinterpret_tensor(buf14, (1, 512, 3072), (1572864, 3072, 1), 0); del buf14  # reuse
        # Source Nodes: [add_23, mul_20], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_8.run(buf18, addmm_22, tanh_5, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_22
        del tanh_5
        buf19 = reinterpret_tensor(buf13, (512, 768), (768, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (512, 3072), (3072, 1), 0), permute_37, out=buf19)
        del permute_37
        buf20 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_38, reinterpret_tensor(buf18, (512, 3072), (3072, 1), 0), out=buf20)
        del permute_38
        buf21 = empty_strided((1, 3072, 4), (12288, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf18, buf21, 12288, 128, grid=grid(12288), stream=stream0)
        buf22 = reinterpret_tensor(buf16, (1, 3072), (3072, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf21, buf22, 3072, 4, grid=grid(3072), stream=stream0)
        buf27 = buf10; del buf10  # reuse
        buf28 = reinterpret_tensor(buf7, (1, 512, 768), (393216, 768, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf27, buf19, primals_73, mul_42, div_15, getitem_75, buf28, 512, 768, grid=grid(512), stream=stream0)
        del div_15
        del getitem_75
        del primals_73
        buf25 = empty((768, ), device='cuda', dtype=torch.float32)
        buf26 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf19, mul_42, buf25, buf26, 768, 512, grid=grid(768), stream=stream0)
        del mul_42
        buf29 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 768), (768, 1), 0), permute_39, out=buf29)
        del permute_39
        buf30 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_40, reinterpret_tensor(buf28, (512, 768), (768, 1), 0), out=buf30)
        del permute_40
        buf31 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf28, buf31, 3072, 128, grid=grid(3072), stream=stream0)
        buf32 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf31, buf32, 768, 4, grid=grid(768), stream=stream0)
        buf33 = reinterpret_tensor(buf28, (12, 512, 64), (32768, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_42, reinterpret_tensor(buf29, (12, 512, 64), (64, 768, 1), 0), out=buf33)
        del permute_42
        buf34 = empty((12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (12, 512, 64), (64, 768, 1), 0), permute_43, out=buf34)
        del permute_43
        buf36 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [full, loss], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.nll_loss_forward, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_12.run(buf34, getitem_73, alias_15, slice_24, buf36, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_15
        del getitem_73
        del slice_24
        buf37 = reinterpret_tensor(buf29, (12, 64, 512), (32768, 512, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_44, reinterpret_tensor(buf36, (12, 512, 512), (262144, 512, 1), 0), out=buf37)
        del permute_44
        buf38 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (12, 512, 512), (262144, 512, 1), 0), permute_45, out=buf38)
        del permute_45
        buf39 = empty((1, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_13.run(buf38, tangents_13, buf37, tangents_14, buf33, buf39, 512, 2304, grid=grid(512, 2304), stream=stream0)
        del tangents_13
        del tangents_14
        buf40 = reinterpret_tensor(buf38, (512, 768), (768, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (512, 2304), (2304, 1), 0), permute_50, out=buf40)
        del permute_50
        buf41 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_51, reinterpret_tensor(buf39, (512, 2304), (2304, 1), 0), out=buf41)
        del permute_51
        buf42 = empty_strided((1, 2304, 4), (9216, 1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf39, buf42, 9216, 128, grid=grid(9216), stream=stream0)
        buf43 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf42, buf43, 2304, 4, grid=grid(2304), stream=stream0)
        buf48 = buf27; del buf27  # reuse
        buf49 = reinterpret_tensor(buf37, (1, 512, 768), (393216, 768, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf48, buf40, primals_71, mul_40, div_17, getitem_66, buf49, 512, 768, grid=grid(512), stream=stream0)
        del div_17
        del getitem_66
        del primals_71
        buf46 = empty((768, ), device='cuda', dtype=torch.float32)
        buf47 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf40, mul_40, buf46, buf47, 768, 512, grid=grid(768), stream=stream0)
        del mul_40
        buf50 = reinterpret_tensor(buf18, (512, 3072), (3072, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (512, 768), (768, 1), 0), permute_52, out=buf50)
        del permute_52
        buf51 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_53, reinterpret_tensor(buf49, (512, 768), (768, 1), 0), out=buf51)
        del permute_53
        buf52 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf49, buf52, 3072, 128, grid=grid(3072), stream=stream0)
        buf53 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf52, buf53, 768, 4, grid=grid(768), stream=stream0)
        buf54 = reinterpret_tensor(buf50, (1, 512, 3072), (1572864, 3072, 1), 0); del buf50  # reuse
        # Source Nodes: [add_19, mul_16], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_8.run(buf54, addmm_18, tanh_4, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_18
        del tanh_4
        buf55 = reinterpret_tensor(buf49, (512, 768), (768, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (512, 3072), (3072, 1), 0), permute_54, out=buf55)
        del permute_54
        buf56 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_55, reinterpret_tensor(buf54, (512, 3072), (3072, 1), 0), out=buf56)
        del permute_55
        buf57 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf54, buf57, 12288, 128, grid=grid(12288), stream=stream0)
        buf58 = reinterpret_tensor(buf52, (1, 3072), (3072, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf57, buf58, 3072, 4, grid=grid(3072), stream=stream0)
        buf63 = buf48; del buf48  # reuse
        buf64 = reinterpret_tensor(buf40, (1, 512, 768), (393216, 768, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf63, buf55, primals_69, mul_34, div_18, getitem_62, buf64, 512, 768, grid=grid(512), stream=stream0)
        del div_18
        del getitem_62
        del primals_69
        buf61 = empty((768, ), device='cuda', dtype=torch.float32)
        buf62 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf55, mul_34, buf61, buf62, 768, 512, grid=grid(768), stream=stream0)
        del mul_34
        buf65 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 768), (768, 1), 0), permute_56, out=buf65)
        del permute_56
        buf66 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_57, reinterpret_tensor(buf64, (512, 768), (768, 1), 0), out=buf66)
        del permute_57
        buf67 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf64, buf67, 3072, 128, grid=grid(3072), stream=stream0)
        buf68 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf67, buf68, 768, 4, grid=grid(768), stream=stream0)
        buf69 = reinterpret_tensor(buf64, (12, 512, 64), (32768, 64, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_59, reinterpret_tensor(buf65, (12, 512, 64), (64, 768, 1), 0), out=buf69)
        del permute_59
        buf70 = reinterpret_tensor(buf36, (12, 512, 512), (262144, 512, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf65, (12, 512, 64), (64, 768, 1), 0), permute_60, out=buf70)
        del permute_60
        buf72 = reinterpret_tensor(buf34, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf34  # reuse
        # Source Nodes: [full, loss], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.nll_loss_forward, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_12.run(buf70, getitem_60, alias_17, slice_20, buf72, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_17
        del getitem_60
        del slice_20
        buf73 = reinterpret_tensor(buf65, (12, 64, 512), (32768, 512, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_61, reinterpret_tensor(buf72, (12, 512, 512), (262144, 512, 1), 0), out=buf73)
        del permute_61
        buf74 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf72, (12, 512, 512), (262144, 512, 1), 0), permute_62, out=buf74)
        del permute_62
        buf75 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_13.run(buf74, tangents_11, buf73, tangents_12, buf69, buf75, 512, 2304, grid=grid(512, 2304), stream=stream0)
        del tangents_11
        del tangents_12
        buf76 = reinterpret_tensor(buf74, (512, 768), (768, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (512, 2304), (2304, 1), 0), permute_67, out=buf76)
        del permute_67
        buf77 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_68, reinterpret_tensor(buf75, (512, 2304), (2304, 1), 0), out=buf77)
        del permute_68
        buf78 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf75, buf78, 9216, 128, grid=grid(9216), stream=stream0)
        buf79 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf78, buf79, 2304, 4, grid=grid(2304), stream=stream0)
        buf84 = buf63; del buf63  # reuse
        buf85 = reinterpret_tensor(buf73, (1, 512, 768), (393216, 768, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf84, buf76, primals_67, mul_32, div_20, getitem_53, buf85, 512, 768, grid=grid(512), stream=stream0)
        del div_20
        del getitem_53
        del primals_67
        buf82 = empty((768, ), device='cuda', dtype=torch.float32)
        buf83 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf76, mul_32, buf82, buf83, 768, 512, grid=grid(768), stream=stream0)
        del mul_32
        buf86 = reinterpret_tensor(buf54, (512, 3072), (3072, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (512, 768), (768, 1), 0), permute_69, out=buf86)
        del permute_69
        buf87 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_70, reinterpret_tensor(buf85, (512, 768), (768, 1), 0), out=buf87)
        del permute_70
        buf88 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf85, buf88, 3072, 128, grid=grid(3072), stream=stream0)
        buf89 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf88, buf89, 768, 4, grid=grid(768), stream=stream0)
        buf90 = reinterpret_tensor(buf86, (1, 512, 3072), (1572864, 3072, 1), 0); del buf86  # reuse
        # Source Nodes: [add_15, mul_12], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_8.run(buf90, addmm_14, tanh_3, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_14
        del tanh_3
        buf91 = reinterpret_tensor(buf85, (512, 768), (768, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (512, 3072), (3072, 1), 0), permute_71, out=buf91)
        del permute_71
        buf92 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_72, reinterpret_tensor(buf90, (512, 3072), (3072, 1), 0), out=buf92)
        del permute_72
        buf93 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf90, buf93, 12288, 128, grid=grid(12288), stream=stream0)
        buf94 = reinterpret_tensor(buf88, (1, 3072), (3072, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf93, buf94, 3072, 4, grid=grid(3072), stream=stream0)
        buf99 = buf84; del buf84  # reuse
        buf100 = reinterpret_tensor(buf76, (1, 512, 768), (393216, 768, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf99, buf91, primals_65, mul_26, div_21, getitem_49, buf100, 512, 768, grid=grid(512), stream=stream0)
        del div_21
        del getitem_49
        del primals_65
        buf97 = empty((768, ), device='cuda', dtype=torch.float32)
        buf98 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf91, mul_26, buf97, buf98, 768, 512, grid=grid(768), stream=stream0)
        del mul_26
        buf101 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (512, 768), (768, 1), 0), permute_73, out=buf101)
        del permute_73
        buf102 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_74, reinterpret_tensor(buf100, (512, 768), (768, 1), 0), out=buf102)
        del permute_74
        buf103 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf100, buf103, 3072, 128, grid=grid(3072), stream=stream0)
        buf104 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf103, buf104, 768, 4, grid=grid(768), stream=stream0)
        buf105 = reinterpret_tensor(buf100, (12, 512, 64), (32768, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_76, reinterpret_tensor(buf101, (12, 512, 64), (64, 768, 1), 0), out=buf105)
        del permute_76
        buf106 = reinterpret_tensor(buf72, (12, 512, 512), (262144, 512, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (12, 512, 64), (64, 768, 1), 0), permute_77, out=buf106)
        del permute_77
        buf108 = reinterpret_tensor(buf70, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf70  # reuse
        # Source Nodes: [full, loss], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.nll_loss_forward, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_12.run(buf106, getitem_47, alias_19, slice_16, buf108, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_19
        del getitem_47
        del slice_16
        buf109 = reinterpret_tensor(buf101, (12, 64, 512), (32768, 512, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_78, reinterpret_tensor(buf108, (12, 512, 512), (262144, 512, 1), 0), out=buf109)
        del permute_78
        buf110 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf108, (12, 512, 512), (262144, 512, 1), 0), permute_79, out=buf110)
        del permute_79
        buf111 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_13.run(buf110, tangents_9, buf109, tangents_10, buf105, buf111, 512, 2304, grid=grid(512, 2304), stream=stream0)
        del tangents_10
        del tangents_9
        buf112 = reinterpret_tensor(buf110, (512, 768), (768, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (512, 2304), (2304, 1), 0), permute_84, out=buf112)
        del permute_84
        buf113 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_85, reinterpret_tensor(buf111, (512, 2304), (2304, 1), 0), out=buf113)
        del permute_85
        buf114 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf111, buf114, 9216, 128, grid=grid(9216), stream=stream0)
        buf115 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf114, buf115, 2304, 4, grid=grid(2304), stream=stream0)
        buf120 = buf99; del buf99  # reuse
        buf121 = reinterpret_tensor(buf109, (1, 512, 768), (393216, 768, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf120, buf112, primals_63, mul_24, div_23, getitem_40, buf121, 512, 768, grid=grid(512), stream=stream0)
        del div_23
        del getitem_40
        del primals_63
        buf118 = empty((768, ), device='cuda', dtype=torch.float32)
        buf119 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf112, mul_24, buf118, buf119, 768, 512, grid=grid(768), stream=stream0)
        del mul_24
        buf122 = reinterpret_tensor(buf90, (512, 3072), (3072, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (512, 768), (768, 1), 0), permute_86, out=buf122)
        del permute_86
        buf123 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_87, reinterpret_tensor(buf121, (512, 768), (768, 1), 0), out=buf123)
        del permute_87
        buf124 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf121, buf124, 3072, 128, grid=grid(3072), stream=stream0)
        buf125 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf124, buf125, 768, 4, grid=grid(768), stream=stream0)
        buf126 = reinterpret_tensor(buf122, (1, 512, 3072), (1572864, 3072, 1), 0); del buf122  # reuse
        # Source Nodes: [add_11, mul_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_8.run(buf126, addmm_10, tanh_2, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_10
        del tanh_2
        buf127 = reinterpret_tensor(buf121, (512, 768), (768, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (512, 3072), (3072, 1), 0), permute_88, out=buf127)
        del permute_88
        buf128 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_89, reinterpret_tensor(buf126, (512, 3072), (3072, 1), 0), out=buf128)
        del permute_89
        buf129 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf126, buf129, 12288, 128, grid=grid(12288), stream=stream0)
        buf130 = reinterpret_tensor(buf124, (1, 3072), (3072, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf129, buf130, 3072, 4, grid=grid(3072), stream=stream0)
        buf135 = buf120; del buf120  # reuse
        buf136 = reinterpret_tensor(buf112, (1, 512, 768), (393216, 768, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf135, buf127, primals_61, mul_18, div_24, getitem_36, buf136, 512, 768, grid=grid(512), stream=stream0)
        del div_24
        del getitem_36
        del primals_61
        buf133 = empty((768, ), device='cuda', dtype=torch.float32)
        buf134 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf127, mul_18, buf133, buf134, 768, 512, grid=grid(768), stream=stream0)
        del mul_18
        buf137 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (512, 768), (768, 1), 0), permute_90, out=buf137)
        del permute_90
        buf138 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_91, reinterpret_tensor(buf136, (512, 768), (768, 1), 0), out=buf138)
        del permute_91
        buf139 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf136, buf139, 3072, 128, grid=grid(3072), stream=stream0)
        buf140 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf139, buf140, 768, 4, grid=grid(768), stream=stream0)
        buf141 = reinterpret_tensor(buf136, (12, 512, 64), (32768, 64, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_93, reinterpret_tensor(buf137, (12, 512, 64), (64, 768, 1), 0), out=buf141)
        del permute_93
        buf142 = reinterpret_tensor(buf108, (12, 512, 512), (262144, 512, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (12, 512, 64), (64, 768, 1), 0), permute_94, out=buf142)
        del permute_94
        buf144 = reinterpret_tensor(buf106, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf106  # reuse
        # Source Nodes: [full, loss], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.nll_loss_forward, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_12.run(buf142, getitem_34, alias_21, slice_12, buf144, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_21
        del getitem_34
        del slice_12
        buf145 = reinterpret_tensor(buf137, (12, 64, 512), (32768, 512, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_95, reinterpret_tensor(buf144, (12, 512, 512), (262144, 512, 1), 0), out=buf145)
        del permute_95
        buf146 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf144, (12, 512, 512), (262144, 512, 1), 0), permute_96, out=buf146)
        del permute_96
        buf147 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_13.run(buf146, tangents_7, buf145, tangents_8, buf141, buf147, 512, 2304, grid=grid(512, 2304), stream=stream0)
        del tangents_7
        del tangents_8
        buf148 = reinterpret_tensor(buf146, (512, 768), (768, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (512, 2304), (2304, 1), 0), permute_101, out=buf148)
        del permute_101
        buf149 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_102, reinterpret_tensor(buf147, (512, 2304), (2304, 1), 0), out=buf149)
        del permute_102
        buf150 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf147, buf150, 9216, 128, grid=grid(9216), stream=stream0)
        buf151 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf150, buf151, 2304, 4, grid=grid(2304), stream=stream0)
        buf156 = buf135; del buf135  # reuse
        buf157 = reinterpret_tensor(buf145, (1, 512, 768), (393216, 768, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf156, buf148, primals_59, mul_16, div_26, getitem_27, buf157, 512, 768, grid=grid(512), stream=stream0)
        del div_26
        del getitem_27
        del primals_59
        buf154 = empty((768, ), device='cuda', dtype=torch.float32)
        buf155 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf148, mul_16, buf154, buf155, 768, 512, grid=grid(768), stream=stream0)
        del mul_16
        buf158 = reinterpret_tensor(buf126, (512, 3072), (3072, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (512, 768), (768, 1), 0), permute_103, out=buf158)
        del permute_103
        buf159 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_104, reinterpret_tensor(buf157, (512, 768), (768, 1), 0), out=buf159)
        del permute_104
        buf160 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf157, buf160, 3072, 128, grid=grid(3072), stream=stream0)
        buf161 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf160, buf161, 768, 4, grid=grid(768), stream=stream0)
        buf162 = reinterpret_tensor(buf158, (1, 512, 3072), (1572864, 3072, 1), 0); del buf158  # reuse
        # Source Nodes: [add_7, mul_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_8.run(buf162, addmm_6, tanh_1, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_6
        del tanh_1
        buf163 = reinterpret_tensor(buf157, (512, 768), (768, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (512, 3072), (3072, 1), 0), permute_105, out=buf163)
        del permute_105
        buf164 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_106, reinterpret_tensor(buf162, (512, 3072), (3072, 1), 0), out=buf164)
        del permute_106
        buf165 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf162, buf165, 12288, 128, grid=grid(12288), stream=stream0)
        buf166 = reinterpret_tensor(buf160, (1, 3072), (3072, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf165, buf166, 3072, 4, grid=grid(3072), stream=stream0)
        buf171 = buf156; del buf156  # reuse
        buf172 = reinterpret_tensor(buf148, (1, 512, 768), (393216, 768, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf171, buf163, primals_57, mul_10, div_27, getitem_23, buf172, 512, 768, grid=grid(512), stream=stream0)
        del div_27
        del getitem_23
        del primals_57
        buf169 = empty((768, ), device='cuda', dtype=torch.float32)
        buf170 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf163, mul_10, buf169, buf170, 768, 512, grid=grid(768), stream=stream0)
        del mul_10
        buf173 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (512, 768), (768, 1), 0), permute_107, out=buf173)
        del permute_107
        buf174 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_108, reinterpret_tensor(buf172, (512, 768), (768, 1), 0), out=buf174)
        del permute_108
        buf175 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf172, buf175, 3072, 128, grid=grid(3072), stream=stream0)
        buf176 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf175, buf176, 768, 4, grid=grid(768), stream=stream0)
        buf177 = reinterpret_tensor(buf172, (12, 512, 64), (32768, 64, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_110, reinterpret_tensor(buf173, (12, 512, 64), (64, 768, 1), 0), out=buf177)
        del permute_110
        buf178 = reinterpret_tensor(buf144, (12, 512, 512), (262144, 512, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (12, 512, 64), (64, 768, 1), 0), permute_111, out=buf178)
        del permute_111
        buf180 = reinterpret_tensor(buf142, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf142  # reuse
        # Source Nodes: [full, loss], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.nll_loss_forward, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_12.run(buf178, getitem_21, alias_23, slice_8, buf180, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_23
        del getitem_21
        del slice_8
        buf181 = reinterpret_tensor(buf173, (12, 64, 512), (32768, 512, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_112, reinterpret_tensor(buf180, (12, 512, 512), (262144, 512, 1), 0), out=buf181)
        del permute_112
        buf182 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (12, 512, 512), (262144, 512, 1), 0), permute_113, out=buf182)
        del permute_113
        buf183 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_13.run(buf182, tangents_5, buf181, tangents_6, buf177, buf183, 512, 2304, grid=grid(512, 2304), stream=stream0)
        del tangents_5
        del tangents_6
        buf184 = reinterpret_tensor(buf182, (512, 768), (768, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (512, 2304), (2304, 1), 0), permute_118, out=buf184)
        del permute_118
        buf185 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_119, reinterpret_tensor(buf183, (512, 2304), (2304, 1), 0), out=buf185)
        del permute_119
        buf186 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf183, buf186, 9216, 128, grid=grid(9216), stream=stream0)
        buf187 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf186, buf187, 2304, 4, grid=grid(2304), stream=stream0)
        buf192 = buf171; del buf171  # reuse
        buf193 = reinterpret_tensor(buf181, (1, 512, 768), (393216, 768, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf192, buf184, primals_55, mul_8, div_29, getitem_14, buf193, 512, 768, grid=grid(512), stream=stream0)
        del div_29
        del getitem_14
        del primals_55
        buf190 = empty((768, ), device='cuda', dtype=torch.float32)
        buf191 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf184, mul_8, buf190, buf191, 768, 512, grid=grid(768), stream=stream0)
        del mul_8
        buf194 = reinterpret_tensor(buf162, (512, 3072), (3072, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (512, 768), (768, 1), 0), permute_120, out=buf194)
        del permute_120
        buf195 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_121, reinterpret_tensor(buf193, (512, 768), (768, 1), 0), out=buf195)
        del permute_121
        buf196 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf193, buf196, 3072, 128, grid=grid(3072), stream=stream0)
        buf197 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf196, buf197, 768, 4, grid=grid(768), stream=stream0)
        buf198 = reinterpret_tensor(buf194, (1, 512, 3072), (1572864, 3072, 1), 0); del buf194  # reuse
        # Source Nodes: [add_3, mul], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_8.run(buf198, addmm_2, tanh, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_2
        del tanh
        buf199 = reinterpret_tensor(buf193, (512, 768), (768, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (512, 3072), (3072, 1), 0), permute_122, out=buf199)
        del permute_122
        buf200 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_123, reinterpret_tensor(buf198, (512, 3072), (3072, 1), 0), out=buf200)
        del permute_123
        buf201 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf198, buf201, 12288, 128, grid=grid(12288), stream=stream0)
        del buf198
        buf202 = reinterpret_tensor(buf196, (1, 3072), (3072, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf201, buf202, 3072, 4, grid=grid(3072), stream=stream0)
        del buf201
        buf207 = buf192; del buf192  # reuse
        buf208 = reinterpret_tensor(buf184, (1, 512, 768), (393216, 768, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf207, buf199, primals_53, mul_2, div_30, getitem_10, buf208, 512, 768, grid=grid(512), stream=stream0)
        del div_30
        del getitem_10
        del primals_53
        buf205 = empty((768, ), device='cuda', dtype=torch.float32)
        buf206 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf199, mul_2, buf205, buf206, 768, 512, grid=grid(768), stream=stream0)
        del mul_2
        buf209 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (512, 768), (768, 1), 0), permute_124, out=buf209)
        del permute_124
        buf210 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_125, reinterpret_tensor(buf208, (512, 768), (768, 1), 0), out=buf210)
        del permute_125
        buf211 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf208, buf211, 3072, 128, grid=grid(3072), stream=stream0)
        buf212 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf211, buf212, 768, 4, grid=grid(768), stream=stream0)
        del buf211
        buf213 = reinterpret_tensor(buf208, (12, 512, 64), (32768, 64, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_127, reinterpret_tensor(buf209, (12, 512, 64), (64, 768, 1), 0), out=buf213)
        del permute_127
        buf214 = reinterpret_tensor(buf180, (12, 512, 512), (262144, 512, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf209, (12, 512, 64), (64, 768, 1), 0), permute_128, out=buf214)
        del permute_128
        buf216 = reinterpret_tensor(buf178, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf178  # reuse
        # Source Nodes: [full, loss], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.nll_loss_forward, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_12.run(buf214, getitem_8, alias_25, slice_4, buf216, 6144, 512, grid=grid(6144), stream=stream0)
        del alias_25
        del buf214
        del getitem_8
        del slice_4
        buf217 = reinterpret_tensor(buf209, (12, 64, 512), (32768, 512, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_129, reinterpret_tensor(buf216, (12, 512, 512), (262144, 512, 1), 0), out=buf217)
        del permute_129
        buf218 = buf177; del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (12, 512, 512), (262144, 512, 1), 0), permute_130, out=buf218)
        del buf216
        del permute_130
        buf219 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_13.run(buf218, tangents_3, buf217, tangents_4, buf213, buf219, 512, 2304, grid=grid(512, 2304), stream=stream0)
        del tangents_3
        del tangents_4
        buf220 = reinterpret_tensor(buf218, (512, 768), (768, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (512, 2304), (2304, 1), 0), permute_135, out=buf220)
        del permute_135
        buf221 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_136, reinterpret_tensor(buf219, (512, 2304), (2304, 1), 0), out=buf221)
        del permute_136
        buf222 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf219, buf222, 9216, 128, grid=grid(9216), stream=stream0)
        del buf219
        buf223 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf222, buf223, 2304, 4, grid=grid(2304), stream=stream0)
        del buf222
        buf228 = buf207; del buf207  # reuse
        buf230 = reinterpret_tensor(buf217, (1, 512, 768), (393216, 768, 1), 0); del buf217  # reuse
        buf234 = reinterpret_tensor(buf213, (1, 512, 768), (393216, 768, 1), 0); del buf213  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_16.run(buf228, buf220, primals_51, mul, div_32, getitem_1, view, buf230, buf234, 512, 768, grid=grid(512), stream=stream0)
        del buf228
        del div_32
        del getitem_1
        del primals_51
        buf226 = empty((768, ), device='cuda', dtype=torch.float32)
        buf227 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf220, mul, buf226, buf227, 768, 512, grid=grid(768), stream=stream0)
        del buf220
        del mul
        buf229 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_17.run(buf229, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf229, [view_1], buf230, True)
        del buf230
        del view_1
        buf233 = empty((50257, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_18.run(buf233, 38597376, grid=grid(38597376), stream=stream0)
        aten.index_put_(buf233, [view], buf234, True)
        del buf234
        del view
        return (reinterpret_tensor(buf223, (2304, ), (1, ), 0), buf221, reinterpret_tensor(buf212, (768, ), (1, ), 0), buf210, reinterpret_tensor(buf202, (3072, ), (1, ), 0), buf200, reinterpret_tensor(buf197, (768, ), (1, ), 0), buf195, reinterpret_tensor(buf187, (2304, ), (1, ), 0), buf185, reinterpret_tensor(buf176, (768, ), (1, ), 0), buf174, reinterpret_tensor(buf166, (3072, ), (1, ), 0), buf164, reinterpret_tensor(buf161, (768, ), (1, ), 0), buf159, reinterpret_tensor(buf151, (2304, ), (1, ), 0), buf149, reinterpret_tensor(buf140, (768, ), (1, ), 0), buf138, reinterpret_tensor(buf130, (3072, ), (1, ), 0), buf128, reinterpret_tensor(buf125, (768, ), (1, ), 0), buf123, reinterpret_tensor(buf115, (2304, ), (1, ), 0), buf113, reinterpret_tensor(buf104, (768, ), (1, ), 0), buf102, reinterpret_tensor(buf94, (3072, ), (1, ), 0), buf92, reinterpret_tensor(buf89, (768, ), (1, ), 0), buf87, reinterpret_tensor(buf79, (2304, ), (1, ), 0), buf77, reinterpret_tensor(buf68, (768, ), (1, ), 0), buf66, reinterpret_tensor(buf58, (3072, ), (1, ), 0), buf56, reinterpret_tensor(buf53, (768, ), (1, ), 0), buf51, reinterpret_tensor(buf43, (2304, ), (1, ), 0), buf41, reinterpret_tensor(buf32, (768, ), (1, ), 0), buf30, reinterpret_tensor(buf22, (3072, ), (1, ), 0), buf20, reinterpret_tensor(buf17, (768, ), (1, ), 0), buf15, buf233, buf229, buf226, buf227, buf205, buf206, buf190, buf191, buf169, buf170, buf154, buf155, buf133, buf134, buf118, buf119, buf97, buf98, buf82, buf83, buf61, buf62, buf46, buf47, buf25, buf26, buf11, buf12, reinterpret_tensor(buf6, (50257, 768), (768, 1), 0), None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    view_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    getitem_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_4 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_8 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_10 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_2 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_8 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_21 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_23 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_10 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_1 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_16 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_12 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_34 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_18 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_2 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_16 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_47 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_49 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_26 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_3 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_32 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_20 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_60 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_62 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_34 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_4 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_66 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_40 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_24 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_73 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_75 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_42 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_5 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_48 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_111 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sub_20 = rand_strided((511, 50257), (50257, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_6 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_33 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_35 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_36 = rand_strided((3072, 512), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_37 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_38 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_39 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_42 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_43 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_44 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_45 = rand_strided((12, 512, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_50 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_51 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_52 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_53 = rand_strided((3072, 512), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_54 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_55 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_56 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_57 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_59 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_60 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_61 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_62 = rand_strided((12, 512, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_67 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_68 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_69 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_70 = rand_strided((3072, 512), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_71 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_72 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_73 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_74 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_76 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_77 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_78 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_79 = rand_strided((12, 512, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_84 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_85 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_86 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_87 = rand_strided((3072, 512), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_88 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_89 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_90 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_91 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_93 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_94 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_95 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_96 = rand_strided((12, 512, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_101 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_102 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_103 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_104 = rand_strided((3072, 512), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_105 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_106 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_107 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_108 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_110 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_111 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_112 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_113 = rand_strided((12, 512, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_118 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_119 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_120 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_121 = rand_strided((3072, 512), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_122 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_124 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_125 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_127 = rand_strided((12, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_128 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_129 = rand_strided((12, 64, 512), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_130 = rand_strided((12, 512, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_135 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_136 = rand_strided((768, 512), (1, 768), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 50257), (25731584, 50257, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_85, view, view_1, getitem_1, mul, slice_4, getitem_8, getitem_10, mul_2, addmm_2, tanh, getitem_14, mul_8, slice_8, getitem_21, getitem_23, mul_10, addmm_6, tanh_1, getitem_27, mul_16, slice_12, getitem_34, getitem_36, mul_18, addmm_10, tanh_2, getitem_40, mul_24, slice_16, getitem_47, getitem_49, mul_26, addmm_14, tanh_3, getitem_53, mul_32, slice_20, getitem_60, getitem_62, mul_34, addmm_18, tanh_4, getitem_66, mul_40, slice_24, getitem_73, getitem_75, mul_42, addmm_22, tanh_5, getitem_79, mul_48, view_111, sub_20, convert_element_type_6, permute_33, div_14, permute_35, permute_36, permute_37, permute_38, div_15, permute_39, permute_40, permute_42, permute_43, alias_15, permute_44, permute_45, permute_50, permute_51, div_17, permute_52, permute_53, permute_54, permute_55, div_18, permute_56, permute_57, permute_59, permute_60, alias_17, permute_61, permute_62, permute_67, permute_68, div_20, permute_69, permute_70, permute_71, permute_72, div_21, permute_73, permute_74, permute_76, permute_77, alias_19, permute_78, permute_79, permute_84, permute_85, div_23, permute_86, permute_87, permute_88, permute_89, div_24, permute_90, permute_91, permute_93, permute_94, alias_21, permute_95, permute_96, permute_101, permute_102, div_26, permute_103, permute_104, permute_105, permute_106, div_27, permute_107, permute_108, permute_110, permute_111, alias_23, permute_112, permute_113, permute_118, permute_119, div_29, permute_120, permute_121, permute_122, permute_123, div_30, permute_124, permute_125, permute_127, permute_128, alias_25, permute_129, permute_130, permute_135, permute_136, div_32, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistillGPT2', benchmark_compiled_module)
