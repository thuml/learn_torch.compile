
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


# kernel path: /tmp/torchinductor_youkaichao/sf/csfpp7x7otrmjprf5spwmld5adgg4fktow2g36rcslgg5erbqhso.py
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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


# kernel path: /tmp/torchinductor_youkaichao/od/cods6s2cxa2iw26v57b4pd3zy6tvey2r7rjjzdxzibzl4obixhib.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wc/cwcfbiktttasxl6bgaslntll63r62g7sasyhccx6btqaglticdts.py
# Source Nodes: [hidden_states_9], Original ATen: [aten.gelu, aten.gelu_backward]
# hidden_states_9 => add_9, erf, mul_9
triton_poi_fused_gelu_gelu_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_2', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/d6/cd6w6jtf2xd4ntdas4qd57pif2dg3qv2fd5mbgy2klptfttgdspm.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuzwtf3qz5zkbjpaacfe4z44w5gxk5xpavq7vjsss537i6wcaxne.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
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
    tmp13 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
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
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g6/cg6gkcboxbtqbsctep6jyjy4l446cg3x5sb3kqrf3radhjb6ihxk.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/et/cetentwkceh3te6dnqg7wluiiwm67uaziqhgqtbjpu3imt5innx2.py
# Source Nodes: [attn_weights_5], Original ATen: [aten._softmax, aten._softmax_backward_data]
# attn_weights_5 => div_1, exp_1, sub_3
triton_per_fused__softmax__softmax_backward_data_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp6 = tmp4 / tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfv4vt3gsoz6lawf6ao3t6kxsf4m6bp4aq7yngik7zjrmojqm7qj.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7t7zugkh6rgagui3wadioto6hwmt2xspdsjyvp5kchsr7c5j62n.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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


# kernel path: /tmp/torchinductor_youkaichao/ti/ctibdgkfzvpds2llgxqtykw3uyexpuojckdllt2kdnpaxaes7ykt.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zp/czpehidlkg74kg5h5oiaqktwlq5er6gmzhmafbvw7rfek53cy5g3.py
# Source Nodes: [], Original ATen: [aten.mul, aten.view]

triton_poi_fused_mul_view_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6yb53ydlw7wg2h2i4fdlyd7g2cy4x3grxct2hg6jfzqdhx44wpg.py
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
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/3u/c3uqxp43jj6tua5qhcyp3e4aaxnf33hsysskfatdcecdxybbu73t.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp8, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33q5dd3ctfur5ilp6u5etjx6dn6h6mxrr46meeheq65lpxsvun5.py
# Source Nodes: [hidden_states], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states => mul, sub
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel):
    xnumel = 128
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
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp6 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 1024.0
    tmp22 = tmp6 * tmp21
    tmp23 = tmp22 - tmp10
    tmp24 = tmp15 * tmp20
    tmp25 = tmp23 - tmp24
    tmp27 = tmp14 / tmp21
    tmp28 = tmp27 * tmp25
    tmp29 = tmp26 + tmp28
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp29, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdnd7rl4eyn5uoezemryq35qmjyer6l7t4bbq2izmurawuanjy4y.py
# Source Nodes: [hidden_states], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states => mul, sub
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_11, primals_21, primals_27, getitem_1, rsqrt, view, view_16, getitem_3, mul_3, view_18, view_20, bmm_2, amax_1, sum_2, view_32, getitem_7, mul_6, view_34, addmm_8, view_36, getitem_11, permute_20, permute_24, div_2, permute_28, permute_33, permute_34, permute_35, permute_36, permute_40, permute_45, permute_49, div_3, permute_53, permute_58, permute_59, alias_3, permute_60, permute_61, permute_65, permute_70, permute_74, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (1024, ), (1, ))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_21, (1024, ), (1, ))
    assert_size_stride(primals_27, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(getitem_1, (1, 128, 1), (128, 1, 1))
    assert_size_stride(rsqrt, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view, (128, 1024), (1024, 1))
    assert_size_stride(view_16, (128, 1024), (1024, 1))
    assert_size_stride(getitem_3, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mul_3, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_18, (128, 1024), (1024, 1))
    assert_size_stride(view_20, (128, 1024), (1024, 1))
    assert_size_stride(bmm_2, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_1, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_2, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_32, (128, 1024), (1024, 1))
    assert_size_stride(getitem_7, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mul_6, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_34, (128, 1024), (1024, 1))
    assert_size_stride(addmm_8, (128, 4096), (4096, 1))
    assert_size_stride(view_36, (128, 4096), (4096, 1))
    assert_size_stride(getitem_11, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(permute_20, (1024, 4096), (4096, 1))
    assert_size_stride(permute_24, (4096, 1024), (1024, 1))
    assert_size_stride(div_2, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_28, (1024, 1024), (1024, 1))
    assert_size_stride(permute_33, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_34, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_35, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_36, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_40, (1024, 1024), (1024, 1))
    assert_size_stride(permute_45, (1024, 1024), (1024, 1))
    assert_size_stride(permute_49, (1024, 1024), (1024, 1))
    assert_size_stride(div_3, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_53, (1024, 1024), (1024, 1))
    assert_size_stride(permute_58, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_59, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_3, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_60, (16, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_61, (16, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_65, (1024, 1024), (1024, 1))
    assert_size_stride(permute_70, (1024, 1024), (1024, 1))
    assert_size_stride(permute_74, (1024, 1024), (1024, 1))
    assert_size_stride(tangents_1, (1, 128, 1024), (131072, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_native_dropout_backward_0.run(tangents_1, getitem_11, buf0, 131072, grid=grid(131072), stream=stream0)
        del getitem_11
        buf1 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (128, 1024), (1024, 1), 0), permute_20, out=buf1)
        del permute_20
        buf2 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1024, 128), (1, 1024), 0), view_36, out=buf2)
        del view_36
        buf3 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf0, buf3, 1024, 128, grid=grid(1024), stream=stream0)
        buf4 = reinterpret_tensor(buf1, (1, 128, 4096), (524288, 4096, 1), 0); del buf1  # reuse
        # Source Nodes: [hidden_states_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_2.run(buf4, addmm_8, 524288, grid=grid(524288), stream=stream0)
        del addmm_8
        buf5 = reinterpret_tensor(buf0, (128, 1024), (1024, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (128, 4096), (4096, 1), 0), permute_24, out=buf5)
        del permute_24
        buf6 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (4096, 128), (1, 4096), 0), view_34, out=buf6)
        del view_34
        buf7 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf4, buf7, 4096, 128, grid=grid(4096), stream=stream0)
        del buf4
        buf12 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf13 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_4.run(buf5, primals_21, mul_6, tangents_1, div_2, getitem_7, buf12, buf13, 128, 1024, grid=grid(128), stream=stream0)
        del div_2
        del getitem_7
        del primals_21
        del tangents_1
        buf10 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf11 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf5, mul_6, buf10, buf11, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_6
        buf14 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (128, 1024), (1024, 1), 0), permute_28, out=buf14)
        del permute_28
        buf15 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1024, 128), (1, 1024), 0), view_32, out=buf15)
        del view_32
        buf16 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf13, buf16, 1024, 128, grid=grid(1024), stream=stream0)
        buf17 = reinterpret_tensor(buf13, (16, 128, 64), (8192, 64, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_33, reinterpret_tensor(buf14, (16, 128, 64), (64, 1024, 1), 0), out=buf17)
        del permute_33
        buf18 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf14, (16, 128, 64), (64, 1024, 1), 0), permute_34, out=buf18)
        del permute_34
        buf20 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_5], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_6.run(buf18, bmm_2, amax_1, sum_2, buf20, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_1
        del bmm_2
        del sum_2
        buf21 = reinterpret_tensor(buf14, (16, 64, 128), (8192, 128, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_35, buf20, out=buf21)
        del permute_35
        buf22 = empty((16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf20, permute_36, out=buf22)
        del permute_36
        buf23 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf17, buf23, 131072, grid=grid(131072), stream=stream0)
        buf24 = reinterpret_tensor(buf17, (128, 1024), (1024, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf23, permute_40, out=buf24)
        del permute_40
        buf25 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1024, 128), (1, 1024), 0), view_20, out=buf25)
        buf26 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf23, buf26, 1024, 128, grid=grid(1024), stream=stream0)
        buf27 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (128, 1024), (1, 128), 0), permute_45, out=buf27)
        del permute_45
        buf28 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (1024, 128), (128, 1), 0), view_20, out=buf28)
        del view_20
        buf29 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf21, buf29, 1024, 128, grid=grid(1024), stream=stream0)
        buf30 = reinterpret_tensor(buf24, (1, 128, 1024), (131072, 1024, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(buf30, buf27, 131072, grid=grid(131072), stream=stream0)
        buf31 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf22, buf31, 131072, grid=grid(131072), stream=stream0)
        buf32 = reinterpret_tensor(buf22, (128, 1024), (1024, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf31, permute_49, out=buf32)
        del permute_49
        buf33 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (1024, 128), (1, 1024), 0), view_18, out=buf33)
        del view_18
        buf34 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf31, buf34, 1024, 128, grid=grid(1024), stream=stream0)
        buf39 = buf12; del buf12  # reuse
        buf40 = reinterpret_tensor(buf31, (1, 128, 1024), (131072, 1024, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf39, buf32, primals_11, mul_3, div_3, getitem_3, buf40, 128, 1024, grid=grid(128), stream=stream0)
        del div_3
        del getitem_3
        del primals_11
        buf37 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf38 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf32, mul_3, buf37, buf38, 1024, 128, grid=grid(1024), stream=stream0)
        del mul_3
        buf41 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (128, 1024), (1024, 1), 0), permute_53, out=buf41)
        del permute_53
        buf42 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (1024, 128), (1, 1024), 0), view_16, out=buf42)
        del view_16
        buf43 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf40, buf43, 1024, 128, grid=grid(1024), stream=stream0)
        buf44 = reinterpret_tensor(buf40, (16, 128, 64), (8192, 64, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_58, reinterpret_tensor(buf41, (16, 128, 64), (64, 1024, 1), 0), out=buf44)
        del permute_58
        buf45 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (16, 128, 64), (64, 1024, 1), 0), permute_59, out=buf45)
        del permute_59
        buf47 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_12.run(buf45, alias_3, buf47, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_3
        del buf45
        buf48 = reinterpret_tensor(buf41, (16, 64, 128), (8192, 128, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_60, reinterpret_tensor(buf47, (16, 128, 128), (16384, 128, 1), 0), out=buf48)
        del permute_60
        buf49 = reinterpret_tensor(buf21, (16, 128, 64), (8192, 64, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf47, (16, 128, 128), (16384, 128, 1), 0), permute_61, out=buf49)
        del buf47
        del permute_61
        buf50 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf44, buf50, 131072, grid=grid(131072), stream=stream0)
        buf51 = reinterpret_tensor(buf44, (128, 1024), (1024, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf50, permute_65, out=buf51)
        del permute_65
        buf52 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (1024, 128), (1, 1024), 0), view, out=buf52)
        buf53 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf50, buf53, 1024, 128, grid=grid(1024), stream=stream0)
        buf54 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (128, 1024), (1, 128), 0), permute_70, out=buf54)
        del permute_70
        buf55 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (1024, 128), (128, 1), 0), view, out=buf55)
        buf56 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf48, buf56, 1024, 128, grid=grid(1024), stream=stream0)
        buf57 = reinterpret_tensor(buf48, (128, 1024), (1024, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf49, buf57, 131072, grid=grid(131072), stream=stream0)
        buf58 = reinterpret_tensor(buf49, (128, 1024), (1024, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf57, permute_74, out=buf58)
        del permute_74
        buf59 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (1024, 128), (1, 1024), 0), view, out=buf59)
        del view
        buf60 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_1.run(buf57, buf60, 1024, 128, grid=grid(1024), stream=stream0)
        del buf57
        buf66 = buf39; del buf39  # reuse
        # Source Nodes: [hidden_states], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13.run(buf66, buf51, buf54, buf58, primals_1, primals_27, getitem_1, rsqrt, 128, 1024, grid=grid(128), stream=stream0)
        del primals_1
        buf64 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf65 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14.run(buf51, buf54, buf58, primals_27, getitem_1, rsqrt, buf64, buf65, 1024, 128, grid=grid(1024), stream=stream0)
        del buf51
        del buf54
        del buf58
        del getitem_1
        del primals_27
        del rsqrt
        return (buf64, buf65, reinterpret_tensor(buf59, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf60, (1024, ), (1, ), 0), reinterpret_tensor(buf55, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf56, (1024, ), (1, ), 0), reinterpret_tensor(buf52, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf53, (1024, ), (1, ), 0), reinterpret_tensor(buf42, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf43, (1024, ), (1, ), 0), buf37, buf38, reinterpret_tensor(buf33, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf34, (1024, ), (1, ), 0), reinterpret_tensor(buf28, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf29, (1024, ), (1, ), 0), reinterpret_tensor(buf25, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf26, (1024, ), (1, ), 0), reinterpret_tensor(buf15, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf16, (1024, ), (1, ), 0), buf10, buf11, reinterpret_tensor(buf6, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf7, (4096, ), (1, ), 0), reinterpret_tensor(buf2, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf3, (1024, ), (1, ), 0), buf66, None, buf30, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_3 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_2 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_1 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_2 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_6 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    permute_20 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_24 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_28 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_33 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_34 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_35 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_36 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_45 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_49 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_53 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_58 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_59 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_3 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_60 = rand_strided((16, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_61 = rand_strided((16, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_65 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_70 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_74 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_11, primals_21, primals_27, getitem_1, rsqrt, view, view_16, getitem_3, mul_3, view_18, view_20, bmm_2, amax_1, sum_2, view_32, getitem_7, mul_6, view_34, addmm_8, view_36, getitem_11, permute_20, permute_24, div_2, permute_28, permute_33, permute_34, permute_35, permute_36, permute_40, permute_45, permute_49, div_3, permute_53, permute_58, permute_59, alias_3, permute_60, permute_61, permute_65, permute_70, permute_74, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('PegasusForConditionalGeneration', benchmark_compiled_module)
