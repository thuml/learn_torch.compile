
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


# kernel path: /tmp/torchinductor_youkaichao/ys/cys3ufcvmf52o7xq4fhoawohis7czb6fuauouijapavr2tjiu6jp.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward]

triton_poi_fused_native_dropout_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x0), tmp5, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/2z/c2zk5ozskraucokolydui7rka5dzf7zws3eck2ved2bndduq274y.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nr/cnrgrp24ygzg4e5gk5kp4upm7mnpgyztqfmhn2k6ncbxzleplc4a.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 16
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkutha6dadz6qxz7somf3l4avp3vkavtdy2pf6uuv7zjtcfuuv2.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tf/ctfvjodapfjivvclirugzv2kvqi66y5vd3ev4z4g7y4zq3wfwnop.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
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


# kernel path: /tmp/torchinductor_youkaichao/zg/czgb3kv3q7vu5el4dmvsudwug67tdtnz4xc4jkenfhjf6pl77ype.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 16
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/az/cazhfxamdan5vtlvi32fhqgqfdqx3abtuwwcnmtkxgofi23fq6re.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp26, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clcgefhgc4wgcfwmu73y5lqeob5gfvo53wiubs7qnvlarocz7t75.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g6/cg6dr3gfyovy6lyhgokbzk5xp4sqidbxvw2gannnzcij62j7tbs4.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.masked_fill, aten.threshold_backward, aten.where]

triton_red_fused__softmax_backward_data_div_masked_fill_threshold_backward_where_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*i1', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_div_masked_fill_threshold_backward_where_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp7 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp8 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp9 * tmp4
        tmp12 = tmp10 - tmp11
        tmp13 = 2.0
        tmp14 = tmp12 / tmp13
        tmp15 = tl.where(tmp7, tmp14, tmp12)
        tmp16 = 0.0
        tmp17 = tl.where(tmp6, tmp16, tmp15)
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp17, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgsvtm26kl3jxhqne4lj4ufix4l6ahexgqnn7ca56egi2p45ii7.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 2048
    x2 = (xindex // 131072)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (64*x2) + (768*x1)), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yz/cyzrivs4shqoqtqujtxhlnvnbrdudkn3mawo5fdq76sfu4ps4vgb.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2048
    y1 = (yindex // 2048)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (2048*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (64*y1) + (768*y0)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/72/c72fjxgu5iuzvvamnocrxojjrfssfltnankajetkyugugnwepwv7.py
# Source Nodes: [], Original ATen: [aten.mul, aten.view]

triton_poi_fused_mul_view_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (131072*(x0 // 64)) + (x0 % 64)), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmn3n2p5edfxfi6jdwbzqmxo5d3v2rrfcmuf6v2iypkiw5sg3gwz.py
# Source Nodes: [hidden_states], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states => mul, sub
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp6 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 768.0
    tmp22 = tmp6 * tmp21
    tmp23 = tmp22 - tmp10
    tmp24 = tmp15 * tmp20
    tmp25 = tmp23 - tmp24
    tmp27 = tmp14 / tmp21
    tmp28 = tmp27 * tmp25
    tmp29 = tmp26 + tmp28
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp29, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6nwribeutmvb7m6bn76kgwqqwryi3yiqjxfaytx6ek5iqxhdayt.py
# Source Nodes: [hidden_states], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states => mul, sub
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp7 = tmp5 - tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp4 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp14 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_11, primals_17, getitem_1, rsqrt, view, view_16, getitem_3, mul_3, add_5, relu, getitem_7, permute_11, permute_15, div_1, permute_19, permute_24, permute_25, alias_3, eq, lt, permute_26, permute_27, permute_31, permute_36, permute_40, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_17, (1, 2048, 768), (1572864, 768, 1))
    assert_size_stride(getitem_1, (1, 2048, 1), (2048, 1, 1))
    assert_size_stride(rsqrt, (1, 2048, 1), (2048, 1, 1))
    assert_size_stride(view, (2048, 768), (768, 1))
    assert_size_stride(view_16, (2048, 768), (768, 1))
    assert_size_stride(getitem_3, (1, 2048, 768), (1572864, 768, 1))
    assert_size_stride(mul_3, (2048, 768), (768, 1))
    assert_size_stride(add_5, (2048, 768), (768, 1))
    assert_size_stride(relu, (2048, 3072), (3072, 1))
    assert_size_stride(getitem_7, (2048, 768), (768, 1))
    assert_size_stride(permute_11, (768, 3072), (3072, 1))
    assert_size_stride(permute_15, (3072, 768), (768, 1))
    assert_size_stride(div_1, (2048, 1), (1, 1))
    assert_size_stride(permute_19, (768, 768), (768, 1))
    assert_size_stride(permute_24, (12, 2048, 2048), (4194304, 1, 2048))
    assert_size_stride(permute_25, (12, 64, 2048), (131072, 1, 64))
    assert_size_stride(alias_3, (12, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(eq, (1, 12, 2048, 2048), (50331648, 4194304, 2048, 1))
    assert_size_stride(lt, (1, 12, 2048, 2048), (50331648, 4194304, 2048, 1))
    assert_size_stride(permute_26, (12, 64, 2048), (131072, 1, 64))
    assert_size_stride(permute_27, (12, 2048, 64), (131072, 64, 1))
    assert_size_stride(permute_31, (768, 768), (768, 1))
    assert_size_stride(permute_36, (768, 768), (768, 1))
    assert_size_stride(permute_40, (768, 768), (768, 1))
    assert_size_stride(tangents_1, (1, 2048, 768), (1572864, 768, 1))
    assert_size_stride(tangents_2, (1, 12, 2048, 64), (1572864, 131072, 64, 1))
    assert_size_stride(tangents_3, (1, 12, 2048, 64), (1572864, 131072, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_native_dropout_backward_0.run(tangents_1, getitem_7, buf0, 1572864, grid=grid(1572864), stream=stream0)
        del getitem_7
        buf1 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, permute_11, out=buf1)
        del permute_11
        buf2 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (768, 2048), (1, 768), 0), relu, out=buf2)
        buf3 = empty_strided((1, 768, 16), (12288, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf0, buf3, 12288, 128, grid=grid(12288), stream=stream0)
        buf4 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_2.run(buf3, buf4, 768, 16, grid=grid(768), stream=stream0)
        buf5 = buf1; del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_3.run(buf5, relu, 6291456, grid=grid(6291456), stream=stream0)
        del relu
        buf6 = buf0; del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf5, permute_15, out=buf6)
        del permute_15
        buf7 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (3072, 2048), (1, 3072), 0), add_5, out=buf7)
        del add_5
        buf8 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf5, buf8, 49152, 128, grid=grid(49152), stream=stream0)
        del buf5
        buf9 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf8, buf9, 3072, 16, grid=grid(3072), stream=stream0)
        del buf8
        buf16 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf17 = empty((1, 2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_6.run(buf6, primals_11, mul_3, tangents_1, div_1, getitem_3, buf16, buf17, 2048, 768, grid=grid(2048), stream=stream0)
        del div_1
        del getitem_3
        del primals_11
        del tangents_1
        buf12 = reinterpret_tensor(buf3, (768, 16), (1, 768), 0); del buf3  # reuse
        buf14 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_7.run(buf6, mul_3, buf12, buf14, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_3
        buf13 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_2.run(buf12, buf13, 768, 16, grid=grid(768), stream=stream0)
        del buf12
        buf15 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_2.run(buf14, buf15, 768, 16, grid=grid(768), stream=stream0)
        buf18 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (2048, 768), (768, 1), 0), permute_19, out=buf18)
        del permute_19
        buf19 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (768, 2048), (1, 768), 0), view_16, out=buf19)
        del view_16
        buf20 = reinterpret_tensor(buf14, (1, 768, 16), (12288, 1, 768), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf17, buf20, 12288, 128, grid=grid(12288), stream=stream0)
        buf21 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_2.run(buf20, buf21, 768, 16, grid=grid(768), stream=stream0)
        buf22 = reinterpret_tensor(buf17, (12, 2048, 64), (131072, 64, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_24, reinterpret_tensor(buf18, (12, 2048, 64), (64, 768, 1), 0), out=buf22)
        del permute_24
        buf23 = empty((12, 2048, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (12, 2048, 64), (64, 768, 1), 0), permute_25, out=buf23)
        del permute_25
        buf25 = empty((1, 12, 2048, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.masked_fill, aten.threshold_backward, aten.where]
        triton_red_fused__softmax_backward_data_div_masked_fill_threshold_backward_where_8.run(buf23, alias_3, lt, eq, buf25, 24576, 2048, grid=grid(24576), stream=stream0)
        del alias_3
        del buf23
        del eq
        del lt
        buf26 = reinterpret_tensor(buf18, (12, 64, 2048), (131072, 2048, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_26, reinterpret_tensor(buf25, (12, 2048, 2048), (4194304, 2048, 1), 0), out=buf26)
        del permute_26
        buf27 = empty((12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf25, (12, 2048, 2048), (4194304, 2048, 1), 0), permute_27, out=buf27)
        del buf25
        del permute_27
        buf28 = empty((1, 2048, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_3, buf22, buf28, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_3
        buf29 = reinterpret_tensor(buf22, (2048, 768), (768, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (2048, 768), (768, 1), 0), permute_31, out=buf29)
        del permute_31
        buf30 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (768, 2048), (1, 768), 0), view, out=buf30)
        buf31 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf28, buf31, 12288, 128, grid=grid(12288), stream=stream0)
        buf32 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_2.run(buf31, buf32, 768, 16, grid=grid(768), stream=stream0)
        buf33 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_2, buf26, buf33, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_2
        buf34 = reinterpret_tensor(buf26, (2048, 768), (768, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (2048, 768), (768, 1), 0), permute_36, out=buf34)
        del permute_36
        buf35 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (768, 2048), (1, 768), 0), view, out=buf35)
        buf36 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf33, buf36, 12288, 128, grid=grid(12288), stream=stream0)
        buf37 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_2.run(buf36, buf37, 768, 16, grid=grid(768), stream=stream0)
        buf38 = reinterpret_tensor(buf33, (2048, 768), (768, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_11.run(buf27, buf38, 1572864, grid=grid(1572864), stream=stream0)
        buf39 = reinterpret_tensor(buf27, (2048, 768), (768, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf38, permute_40, out=buf39)
        del permute_40
        buf40 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (768, 2048), (1, 768), 0), view, out=buf40)
        del view
        buf41 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf38, buf41, 12288, 128, grid=grid(12288), stream=stream0)
        del buf38
        buf42 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_2.run(buf41, buf42, 768, 16, grid=grid(768), stream=stream0)
        del buf41
        buf48 = reinterpret_tensor(buf16, (1, 2048, 768), (1572864, 768, 1), 0); del buf16  # reuse
        # Source Nodes: [hidden_states], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf48, buf29, buf34, buf39, primals_1, primals_17, getitem_1, rsqrt, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_1
        buf46 = empty((768, ), device='cuda', dtype=torch.float32)
        buf47 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13.run(buf29, buf34, buf39, primals_17, getitem_1, rsqrt, buf46, buf47, 768, 2048, grid=grid(768), stream=stream0)
        del buf29
        del buf34
        del buf39
        del getitem_1
        del primals_17
        del rsqrt
        return (buf46, buf47, reinterpret_tensor(buf40, (768, 768), (768, 1), 0), reinterpret_tensor(buf42, (768, ), (1, ), 0), reinterpret_tensor(buf35, (768, 768), (768, 1), 0), reinterpret_tensor(buf37, (768, ), (1, ), 0), reinterpret_tensor(buf30, (768, 768), (768, 1), 0), reinterpret_tensor(buf32, (768, ), (1, ), 0), reinterpret_tensor(buf19, (768, 768), (768, 1), 0), reinterpret_tensor(buf21, (768, ), (1, ), 0), buf13, buf15, reinterpret_tensor(buf7, (3072, 768), (768, 1), 0), reinterpret_tensor(buf9, (3072, ), (1, ), 0), reinterpret_tensor(buf2, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf4, (768, ), (1, ), 0), buf48, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((1, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((1, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_3 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    add_5 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.bool)
    permute_11 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_15 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((2048, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    permute_19 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_24 = rand_strided((12, 2048, 2048), (4194304, 1, 2048), device='cuda:0', dtype=torch.float32)
    permute_25 = rand_strided((12, 64, 2048), (131072, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_3 = rand_strided((12, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float32)
    eq = rand_strided((1, 12, 2048, 2048), (50331648, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    lt = rand_strided((1, 12, 2048, 2048), (50331648, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_26 = rand_strided((12, 64, 2048), (131072, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_27 = rand_strided((12, 2048, 64), (131072, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_31 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_36 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 12, 2048, 64), (1572864, 131072, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 12, 2048, 64), (1572864, 131072, 64, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_11, primals_17, getitem_1, rsqrt, view, view_16, getitem_3, mul_3, add_5, relu, getitem_7, permute_11, permute_15, div_1, permute_19, permute_24, permute_25, alias_3, eq, lt, permute_26, permute_27, permute_31, permute_36, permute_40, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('OPTForCausalLM', benchmark_compiled_module)
