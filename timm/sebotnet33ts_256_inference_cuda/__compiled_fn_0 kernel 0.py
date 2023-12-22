
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


# kernel path: /tmp/torchinductor_youkaichao/rr/crrwkxcpboofanir4ahdijl45q63dz5oiulxumeqxm5cu3ocy5kg.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (196608*y1)), tmp0, ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/rg/crgkwpletxoczwxntyy3ztnvadsvyq2czque4m2hp7ckch375fpa.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 72
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctn2gacunxkoiqexg5j7u5nzau4hklkn3yvenvswnwtmm3dmu54.py
# Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_1 => add_1, mul_1, mul_2, sub
# x_4 => mul_3, sigmoid
triton_poi_fused__native_batch_norm_legit_no_training_silu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (24*x2) + (393216*y1)), tmp16, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ka/ckarrdkilcyebxmwlpkogjmlmf2uw5zdpjhg5ts3552c5haul5x2.py
# Source Nodes: [x_4, x_5], Original ATen: [aten.convolution, aten.silu]
# x_4 => mul_3, sigmoid
# x_5 => convolution_1
triton_poi_fused_convolution_silu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (216*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camdgfegwgo7wca7icnicef2ujifxhrye5triela55fvcko4petv.py
# Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_6 => add_3, mul_5, mul_6, sub_1
# x_9 => mul_7, sigmoid_1
triton_poi_fused__native_batch_norm_legit_no_training_silu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (32*x2) + (524288*y1)), tmp16, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3pjz3gwujyv36jheavnjqzkcaxilgaemcolvzfazaux57m7p6ra.py
# Source Nodes: [x_10, x_9], Original ATen: [aten.convolution, aten.silu]
# x_10 => convolution_2
# x_9 => mul_7, sigmoid_1
triton_poi_fused_convolution_silu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77igfpob2wnls5343u5hg4hkrvcfeel2p3lr6mi2ow2rxuoqczf.py
# Source Nodes: [shortcut, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# shortcut => mul_11, sigmoid_2
# x_11 => add_5, mul_10, mul_9, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_silu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp16, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ek/cekwwvp7hwg6sbv2xeefawyoco3viboqbw35koxahb5uvfsjqy2n.py
# Source Nodes: [x_21, x_22], Original ATen: [aten.convolution, aten.silu]
# x_21 => mul_15, sigmoid_3
# x_22 => convolution_4
triton_poi_fused_convolution_silu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqideljcurosqucfcczdassleq32gzgrere24jtoowhouixyhclj.py
# Source Nodes: [x_23, x_27, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_23 => add_9, mul_17, mul_18, sub_4
# x_27 => mul_19, sigmoid_4
# x_se => mean
triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 64
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (4096*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 1e-05
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt(tmp5)
        tmp7 = 1 / tmp6
        tmp8 = 1.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp2 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tl.sigmoid(tmp14)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tl.store(in_out_ptr0 + (r2 + (4096*x3)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = 4096.0
    tmp21 = tmp18 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gt/cgt7hphuh4ac4w7vc2th35awrccna6ivy4nmw3jo6xltjnupf3og.py
# Source Nodes: [x_27, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
# x_27 => mul_19, sigmoid_4
# x_se => mean
# x_se_1 => convolution_5
# x_se_2 => relu
triton_poi_fused_convolution_mean_relu_silu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_silu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wnij4jgmgxamclveuzmkwmzld5gjntx4hweoygjqoa6sli47vv.py
# Source Nodes: [sigmoid, x_27, x_29, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# sigmoid => sigmoid_5
# x_27 => mul_19, sigmoid_4
# x_29 => mul_20
# x_se => mean
# x_se_1 => convolution_5
# x_se_2 => relu
# x_se_3 => convolution_6
triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp7, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/csks6dkiqjz3mlkkublt3onub3reytpzwundsqnbblvtcwsnbi6r.py
# Source Nodes: [shortcut_1, x_31, x_39, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_1 => mul_27, sigmoid_6
# x_31 => add_11, mul_22, mul_23, sub_5
# x_39 => add_13, mul_25, mul_26, sub_6
# x_43 => add_14
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr0 + (y0 + (256*x2) + (1048576*y1)), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csagizq2od3rxpp5s2ducoj3e4cyk6zxyxxp4nc4sw7kz6hhwwaa.py
# Source Nodes: [shortcut_2, x_59, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_2 => mul_40, sigmoid_10
# x_59 => add_20, mul_38, mul_39, sub_9
# x_66 => add_21
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (y0 + (256*x2) + (1048576*y1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + (y0 + (256*x2) + (1048576*y1)), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ut/cutkbaml5fjmhgkivdjocaxsqxffr6enm5l4mp6cm6tfcbfl3bns.py
# Source Nodes: [x_68, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_68 => add_23, mul_42, mul_43, sub_10
# x_72 => mul_44, sigmoid_11
triton_poi_fused__native_batch_norm_legit_no_training_silu_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (128*x2) + (524288*y1)), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cpr6xwkrd2wzvs4ldytjeslc6v47ggngsywbp2ttjhz2sgv6f3cv.py
# Source Nodes: [x_72, x_73], Original ATen: [aten.convolution, aten.silu]
# x_72 => mul_44, sigmoid_11
# x_73 => convolution_15
triton_poi_fused_convolution_silu_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/45/c45wlwryijgy5rq5mrikwyq2kc4acjibdwrz5wodz37nmtpj6y64.py
# Source Nodes: [x_74, x_78, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_74 => add_25, mul_46, mul_47, sub_11
# x_78 => mul_48, sigmoid_12
# x_se_8 => mean_2
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 1024.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (1024*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/cleicoyuaa7bcwsqp6wvv3zaoqbjor2rxp2zhcqsf326maftyqgc.py
# Source Nodes: [sigmoid_2, x_78, x_80, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# sigmoid_2 => sigmoid_13
# x_78 => mul_48, sigmoid_12
# x_80 => mul_49
# x_se_10 => relu_2
# x_se_11 => convolution_17
# x_se_8 => mean_2
# x_se_9 => convolution_16
triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask)
    tmp3 = tl.load(in_ptr1 + (y3), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (128*x2) + (131072*y1)), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jz/cjz3jkbb2mmuiim77nuhq7kkqemtzpsnw4frdnezlhrko7lc2tb7.py
# Source Nodes: [shortcut_3, x_82, x_90, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_3 => mul_56, sigmoid_14
# x_82 => add_27, mul_51, mul_52, sub_12
# x_90 => add_29, mul_54, mul_55, sub_13
# x_94 => add_30
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr0 + (y0 + (512*x2) + (524288*y1)), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oo/coop4kivhvdnb6i7kg3l4rncebxqvwmlctl7szomjthdsrcljgev.py
# Source Nodes: [x_100, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_100 => mul_60, sigmoid_15
# x_96 => add_32, mul_58, mul_59, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_silu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask)
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (128*x2) + (131072*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ho/chohoxcjpjp3rlctul5xp2vydf2hwkveukouacieg4mrkstekpu3.py
# Source Nodes: [shortcut_4, x_110, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_4 => mul_69, sigmoid_18
# x_110 => add_36, mul_67, mul_68, sub_16
# x_117 => add_37
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + (y0 + (512*x2) + (524288*y1)), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pl/cplht2mkwaml2gwjqfk2zh4kslgm73y3tcwxxkhhs6f6ch53rihy.py
# Source Nodes: [reshape], Original ATen: [aten.clone]
# reshape => clone
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 131072
    x1 = (xindex // 131072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (393216*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/po/cpoexhv6wgyoz6zvincwh4ngdkl6yjqgkbymfpjhdqumyr5ta5yi.py
# Source Nodes: [k_1], Original ATen: [aten.clone]
# k_1 => clone_1
triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 131072
    x1 = (xindex // 131072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (131072 + x0 + (393216*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdnev3zypnwyqfc2snnnpreclmzdkcl3lat4bsz2m7rswxkwag7t.py
# Source Nodes: [x_130], Original ATen: [aten.clone]
# x_130 => clone_4
triton_poi_fused_clone_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 32
    x3 = (xindex // 32)
    y0 = yindex % 32
    y1 = (yindex // 32)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x3) + (1024*x2) + (1024*((y0 + (32*x3)) // 1024)) + (32768*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (1024*y4)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/cscbvvxjvafujo2y4gvan2ujsgs66i46lifaq5wivhzcxggkxiw7.py
# Source Nodes: [x_126], Original ATen: [aten.clone]
# x_126 => clone_3
triton_poi_fused_clone_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/go/cgos4mlgl3iszhjsmhqnwff2qfytzhatrw23crkecf3x3cyk6pw2.py
# Source Nodes: [attn, attn_1, mul_4], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn => add_41
# attn_1 => amax, div, exp, sub_18, sum_1
# mul_4 => mul_74
triton_red_fused__softmax_add_mul_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.1767766952966369
        tmp2 = tmp0 * tmp1
        tmp3 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp4 = tl.full([1, 1], 2048, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp7 = tl.full([1, 1], 63, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp5, tmp12, tmp13)
        tmp15 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp16 = tmp15 < tmp4
        tmp17 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp18 = tmp17 < tmp7
        tmp19 = tmp18 & tmp16
        tmp20 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp16, tmp22, tmp23)
        tmp25 = tmp14 + tmp24
        tmp26 = tmp2 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp30 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = 0.1767766952966369
        tmp32 = tmp30 * tmp31
        tmp33 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp34 = tl.full([1, 1], 2048, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp37 = tl.full([1, 1], 63, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp39, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
        tmp42 = tl.where(tmp39, tmp40, tmp41)
        tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
        tmp44 = tl.where(tmp35, tmp42, tmp43)
        tmp45 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp46 = tmp45 < tmp34
        tmp47 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp48 = tmp47 < tmp37
        tmp49 = tmp48 & tmp46
        tmp50 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp49, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
        tmp52 = tl.where(tmp49, tmp50, tmp51)
        tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
        tmp54 = tl.where(tmp46, tmp52, tmp53)
        tmp55 = tmp44 + tmp54
        tmp56 = tmp32 + tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(rmask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp62 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = 0.1767766952966369
        tmp64 = tmp62 * tmp63
        tmp65 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp66 = tl.full([1, 1], 2048, tl.int64)
        tmp67 = tmp65 < tmp66
        tmp68 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp69 = tl.full([1, 1], 63, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tmp70 & tmp67
        tmp72 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp71, tmp72, tmp73)
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp67, tmp74, tmp75)
        tmp77 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp78 = tmp77 < tmp66
        tmp79 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp80 = tmp79 < tmp69
        tmp81 = tmp80 & tmp78
        tmp82 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
        tmp84 = tl.where(tmp81, tmp82, tmp83)
        tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
        tmp86 = tl.where(tmp78, tmp84, tmp85)
        tmp87 = tmp76 + tmp86
        tmp88 = tmp64 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl.exp(tmp89)
        tmp91 = tmp90 / tmp60
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp91, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3m/c3mt6b5f2vvxutmnu4xdx3r5w7fqv5zq6mbhfn6bogr2i4tnltgu.py
# Source Nodes: [reshape_2], Original ATen: [aten.clone]
# reshape_2 => clone_2
triton_poi_fused_clone_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 131072
    x1 = (xindex // 131072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (262144 + x0 + (393216*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4e/c4eqmif32qhsgw2cvvadq7uro33o3el4k7dmfvav64l4ra2go4pp.py
# Source Nodes: [x_135, x_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_135 => add_43, mul_76, mul_77, sub_19
# x_138 => mul_78, sigmoid_20
triton_poi_fused__native_batch_norm_legit_no_training_silu_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 1024
    x2 = (xindex // 131072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (32768*((x1 + (1024*x0)) // 32768)) + (131072*x2) + (x0 % 32)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzqcttafdgrascnes3ykqotigdgymgiavf2mqzmfbm6gacs2ywy.py
# Source Nodes: [x_148, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_148 => add_48, mul_84, mul_85, sub_21
# x_152 => mul_86, sigmoid_22
triton_poi_fused__native_batch_norm_legit_no_training_silu_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (256*x2) + (262144*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hj/chjpksustsvmkfkolt4p6c43l5v6vrqpdhe4ycbzauxlexstfyw4.py
# Source Nodes: [x_152, x_153], Original ATen: [aten.convolution, aten.silu]
# x_152 => mul_86, sigmoid_22
# x_153 => convolution_29
triton_poi_fused_convolution_silu_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bk/cbk5wrxcti3woaurbrymco7vcf63esuyw7lzxsdnfcstwbb3kngt.py
# Source Nodes: [x_154, x_158, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_154 => add_50, mul_88, mul_89, sub_22
# x_158 => mul_90, sigmoid_23
# x_se_16 => mean_4
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 256.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp14, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctjoem4ssop74wcrt7aukovoqepb4vesztgn7nr363v73qyg7ul.py
# Source Nodes: [x_158, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
# x_158 => mul_90, sigmoid_23
# x_se_16 => mean_4
# x_se_17 => convolution_30
# x_se_18 => relu_4
triton_poi_fused_convolution_mean_relu_silu_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_silu_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3owpqqomuwozmss3h3uqvp43o6thjwzmwmzgbcxlscchl3ylzl.py
# Source Nodes: [sigmoid_4, x_158, x_160, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# sigmoid_4 => sigmoid_24
# x_158 => mul_90, sigmoid_23
# x_160 => mul_91
# x_se_16 => mean_4
# x_se_17 => convolution_30
# x_se_18 => relu_4
# x_se_19 => convolution_31
triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (256*x2) + (65536*y1)), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2efqvmdcpiwkebrtn2azjqcjo2bntlfo7la52oviwn7r2hoonp3.py
# Source Nodes: [shortcut_6, x_162, x_170, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_6 => mul_98, sigmoid_25
# x_162 => add_52, mul_93, mul_94, sub_23
# x_170 => add_54, mul_96, mul_97, sub_24
# x_174 => add_55
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr0 + (y0 + (1024*x2) + (262144*y1)), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/con7yvgqlpji5qqc4ivix2pytfollxon22v3f5wcoybeipozysal.py
# Source Nodes: [x_176, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_176 => add_57, mul_100, mul_101, sub_25
# x_180 => mul_102, sigmoid_26
triton_poi_fused__native_batch_norm_legit_no_training_silu_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (256*x2) + (65536*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/ckiix7sbmscolbeopvrulgbielc3467rbjftsbdsjyi5qj7n5mvk.py
# Source Nodes: [shortcut_7, x_190, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_7 => mul_111, sigmoid_29
# x_190 => add_61, mul_109, mul_110, sub_27
# x_197 => add_62
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + (y0 + (1024*x2) + (262144*y1)), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tr/ctr7ii3yfy4kwhugciz2fgdoxw4ghjjssek4gv4utsn766pnjmrm.py
# Source Nodes: [reshape_12], Original ATen: [aten.clone]
# reshape_12 => clone_7
triton_poi_fused_clone_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 65536
    x1 = (xindex // 65536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196608*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lj/cljabzzlm2hzkbkqgs2qcjirk4q6at3qj7brlcx3fo55qwlljet6.py
# Source Nodes: [k_3], Original ATen: [aten.clone]
# k_3 => clone_8
triton_poi_fused_clone_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 65536
    x1 = (xindex // 65536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (65536 + x0 + (196608*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chk6vorg5iispm5tn3o3ur4ev3ugaumf4a3dd3bc3wwpy6k235cg.py
# Source Nodes: [x_210], Original ATen: [aten.clone]
# x_210 => clone_11
triton_poi_fused_clone_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y0 = yindex % 16
    y1 = (yindex // 16)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x3) + (256*x2) + (256*((y0 + (16*x3)) // 256)) + (16384*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (1024*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyywg77ufwh25k5eadcss3bnnkqavq5fxhvurojcay4tx2dbg7y.py
# Source Nodes: [x_206], Original ATen: [aten.clone]
# x_206 => clone_10
triton_poi_fused_clone_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (16384*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k7/ck75y2fbxjylvjjx45xvglsbrkaynne55iflugemegalcley73zc.py
# Source Nodes: [attn_2, attn_3, mul_7], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_2 => add_66
# attn_3 => amax_1, div_1, exp_1, sub_29, sum_2
# mul_7 => mul_116
triton_red_fused__softmax_add_mul_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.125
        tmp2 = tmp0 * tmp1
        tmp3 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp4 = tl.full([1, 1], 512, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp7 = tl.full([1, 1], 31, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp5, tmp12, tmp13)
        tmp15 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp16 = tmp15 < tmp4
        tmp17 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp18 = tmp17 < tmp7
        tmp19 = tmp18 & tmp16
        tmp20 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp16, tmp22, tmp23)
        tmp25 = tmp14 + tmp24
        tmp26 = tmp2 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp30 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = 0.125
        tmp32 = tmp30 * tmp31
        tmp33 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp34 = tl.full([1, 1], 512, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp37 = tl.full([1, 1], 31, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp39, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
        tmp42 = tl.where(tmp39, tmp40, tmp41)
        tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
        tmp44 = tl.where(tmp35, tmp42, tmp43)
        tmp45 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp46 = tmp45 < tmp34
        tmp47 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp48 = tmp47 < tmp37
        tmp49 = tmp48 & tmp46
        tmp50 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp49, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
        tmp52 = tl.where(tmp49, tmp50, tmp51)
        tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
        tmp54 = tl.where(tmp46, tmp52, tmp53)
        tmp55 = tmp44 + tmp54
        tmp56 = tmp32 + tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(rmask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp62 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = 0.125
        tmp64 = tmp62 * tmp63
        tmp65 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp66 = tl.full([1, 1], 512, tl.int64)
        tmp67 = tmp65 < tmp66
        tmp68 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp69 = tl.full([1, 1], 31, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tmp70 & tmp67
        tmp72 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp71, tmp72, tmp73)
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp67, tmp74, tmp75)
        tmp77 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp78 = tmp77 < tmp66
        tmp79 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp80 = tmp79 < tmp69
        tmp81 = tmp80 & tmp78
        tmp82 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
        tmp84 = tl.where(tmp81, tmp82, tmp83)
        tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
        tmp86 = tl.where(tmp78, tmp84, tmp85)
        tmp87 = tmp76 + tmp86
        tmp88 = tmp64 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl.exp(tmp89)
        tmp91 = tmp90 / tmp60
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp91, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7eqsr35nt7chsbk4kebbfaep4gzxttjpe2ovx47kevs67u5b2pv.py
# Source Nodes: [reshape_14], Original ATen: [aten.clone]
# reshape_14 => clone_9
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 65536
    x1 = (xindex // 65536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (131072 + x0 + (196608*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ks/cks2xccioi4hbx6we7gpfruzdfs6wt7m6txl4pts657o2sfq24uo.py
# Source Nodes: [x_215, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_215 => add_68, mul_118, mul_119, sub_30
# x_218 => mul_120, sigmoid_31
triton_poi_fused__native_batch_norm_legit_no_training_silu_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 256
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (16384*((x1 + (256*x0)) // 16384)) + (65536*x2) + (x0 % 64)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpk5djzxxfqwrxb2cvqq5ac5fr2c27ime533jam3ixuny6buoo2x.py
# Source Nodes: [x_228, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_228 => add_73, mul_126, mul_127, sub_32
# x_232 => mul_128, sigmoid_33
triton_poi_fused__native_batch_norm_legit_no_training_silu_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (512*x2) + (131072*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfag5njvtga2peqaic6lqouht57lasck7msnadbzkz64ubtmjqc.py
# Source Nodes: [x_239], Original ATen: [aten.clone]
# x_239 => clone_18
triton_poi_fused_clone_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 128
    x3 = (xindex // 128)
    y0 = yindex % 16
    y1 = (yindex // 16)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x3) + (256*x2) + (256*((y0 + (16*x3)) // 256)) + (32768*y1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (2048*y4)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyyd5en5tlxr37yzlgatvj4ij745kbed6j7wk6fxg5tptcnagyvk.py
# Source Nodes: [x_235], Original ATen: [aten.clone]
# x_235 => clone_17
triton_poi_fused_clone_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rd/crd7ytmftv6djnkswc7p6h7swnvlnn6jpq74rs3yifto3mt6b4pu.py
# Source Nodes: [attn_4, attn_5, mul_8], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_4 => add_75
# attn_5 => amax_2, div_2, exp_2, sub_33, sum_3
# mul_8 => mul_129
triton_red_fused__softmax_add_mul_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.08838834764831845
        tmp2 = tmp0 * tmp1
        tmp3 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp4 = tl.full([1, 1], 512, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp7 = tl.full([1, 1], 31, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp5, tmp12, tmp13)
        tmp15 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp16 = tmp15 < tmp4
        tmp17 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp18 = tmp17 < tmp7
        tmp19 = tmp18 & tmp16
        tmp20 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp16, tmp22, tmp23)
        tmp25 = tmp14 + tmp24
        tmp26 = tmp2 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp30 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = 0.08838834764831845
        tmp32 = tmp30 * tmp31
        tmp33 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp34 = tl.full([1, 1], 512, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp37 = tl.full([1, 1], 31, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp39, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
        tmp42 = tl.where(tmp39, tmp40, tmp41)
        tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
        tmp44 = tl.where(tmp35, tmp42, tmp43)
        tmp45 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp46 = tmp45 < tmp34
        tmp47 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp48 = tmp47 < tmp37
        tmp49 = tmp48 & tmp46
        tmp50 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp49, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
        tmp52 = tl.where(tmp49, tmp50, tmp51)
        tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
        tmp54 = tl.where(tmp46, tmp52, tmp53)
        tmp55 = tmp44 + tmp54
        tmp56 = tmp32 + tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(rmask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp62 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = 0.08838834764831845
        tmp64 = tmp62 * tmp63
        tmp65 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp66 = tl.full([1, 1], 512, tl.int64)
        tmp67 = tmp65 < tmp66
        tmp68 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp69 = tl.full([1, 1], 31, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tmp70 & tmp67
        tmp72 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp71, tmp72, tmp73)
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp67, tmp74, tmp75)
        tmp77 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp78 = tmp77 < tmp66
        tmp79 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp80 = tmp79 < tmp69
        tmp81 = tmp80 & tmp78
        tmp82 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
        tmp84 = tl.where(tmp81, tmp82, tmp83)
        tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
        tmp86 = tl.where(tmp78, tmp84, tmp85)
        tmp87 = tmp76 + tmp86
        tmp88 = tmp64 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl.exp(tmp89)
        tmp91 = tmp90 / tmp60
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp91, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5a/c5axc6eipz5yvpthsp7vykcxg4afkbof3vaghrq434utocrm3w2g.py
# Source Nodes: [x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d]
# x_243 => avg_pool2d
# x_244 => add_77, mul_131, mul_132, sub_34
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 8
    x2 = (xindex // 4096) % 8
    x3 = (xindex // 32768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*x1) + (4096*x2) + (32768*((x1 + (16*x2) + (128*x0)) // 16384)) + (131072*x3) + (((x1 + (16*x2) + (128*x0)) // 128) % 128)), None)
    tmp1 = tl.load(in_ptr0 + (128 + (256*x1) + (4096*x2) + (32768*((1 + (2*x1) + (32*x2) + (256*x0)) // 32768)) + (131072*x3) + (((1 + (2*x1) + (32*x2) + (256*x0)) // 256) % 128)), None)
    tmp3 = tl.load(in_ptr0 + (2048 + (256*x1) + (4096*x2) + (32768*((8 + x1 + (16*x2) + (128*x0)) // 16384)) + (131072*x3) + (((8 + x1 + (16*x2) + (128*x0)) // 128) % 128)), None)
    tmp5 = tl.load(in_ptr0 + (2176 + (256*x1) + (4096*x2) + (32768*((17 + (2*x1) + (32*x2) + (256*x0)) // 32768)) + (131072*x3) + (((17 + (2*x1) + (32*x2) + (256*x0)) // 256) % 128)), None)
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr0 + (x4), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxff76wafljfut7juihynoxstoh6omcce3mje7gqkigzaqvg3y6w.py
# Source Nodes: [x_247], Original ATen: [aten.silu]
# x_247 => mul_133, sigmoid_34
triton_poi_fused_silu_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1, 2))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (y0 + (512*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (512*x2) + (32768*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7m4lfcgka7cjigpyxi6ekc7uhgouw2iehhzgezmjb7a7uq5edq.py
# Source Nodes: [shortcut_9, x_249, x_256, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_9 => mul_140, sigmoid_35
# x_249 => add_79, mul_135, mul_136, sub_35
# x_256 => add_81, mul_138, mul_139, sub_36
# x_260 => add_82
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1536
    y1 = (yindex // 1536)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr0 + (y0 + (1536*x2) + (98304*y1)), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lf/clfbls25nmqq7343tw6b5rbjptb426hvqiybgxbwviy6rwglcf3e.py
# Source Nodes: [x_262, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_262 => add_84, mul_142, mul_143, sub_37
# x_266 => mul_144, sigmoid_36
triton_poi_fused__native_batch_norm_legit_no_training_silu_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (512*x2) + (32768*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpe3ib5zy74cfu7j2d2ddpajrqrf57iabf6g6ufkckdfirvlnvsc.py
# Source Nodes: [reshape_36], Original ATen: [aten.clone]
# reshape_36 => clone_21
triton_poi_fused_clone_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = (xindex // 32768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (98304*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5z/c5zjlt573bgoe2g63vivxmknswwsmbpzfk4hvvba6p7nlnraqem3.py
# Source Nodes: [k_7], Original ATen: [aten.clone]
# k_7 => clone_22
triton_poi_fused_clone_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = (xindex // 32768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (32768 + x0 + (98304*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5o/c5odrq4zkle74uqdwdfatoyjipdejpmtmqugskibt4zulgmzxqkv.py
# Source Nodes: [x_273], Original ATen: [aten.clone]
# x_273 => clone_25
triton_poi_fused_clone_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 128
    x3 = (xindex // 128)
    y0 = yindex % 8
    y1 = (yindex // 8)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (8*x3) + (64*x2) + (64*((y0 + (8*x3)) // 64)) + (8192*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (1024*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b5/cb52co7mrqemh3fjwr75iq2jcqfnwfi3yafzkblkvtpbv3lnlfso.py
# Source Nodes: [x_269], Original ATen: [aten.clone]
# x_269 => clone_24
triton_poi_fused_clone_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (8192*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c5/cc5j6bxjtlzqp53q7dwtqkh4ngdhkak62m2e75gryfdp26ip237y.py
# Source Nodes: [attn_6, attn_7, mul_9], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_6 => add_86
# attn_7 => amax_3, div_3, exp_3, sub_38, sum_4
# mul_9 => mul_145
triton_per_fused__softmax_add_mul_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = 0.08838834764831845
    tmp2 = tmp0 * tmp1
    tmp3 = 7 + (15*(x0 // 8)) + (r2 // 8)
    tmp4 = tl.full([1, 1], 128, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = (7 + (15*(x0 // 8)) + (r2 // 8)) % 16
    tmp7 = tl.full([1, 1], 15, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + ((15*((7 + (15*(x0 // 8)) + (r2 // 8)) // 16)) + (120*(x0 % 8)) + (960*x1) + ((7 + (15*(x0 // 8)) + (r2 // 8)) % 16)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = 7 + (15*(x0 % 8)) + (r2 % 8)
    tmp16 = tmp15 < tmp4
    tmp17 = (7 + (15*(x0 % 8)) + (r2 % 8)) % 16
    tmp18 = tmp17 < tmp7
    tmp19 = tmp18 & tmp16
    tmp20 = tl.load(in_ptr2 + ((15*(((7 + (15*(x0 % 8)) + (r2 % 8)) // 16) % 8)) + (120*(x0 // 8)) + (960*x1) + ((7 + (15*(x0 % 8)) + (r2 % 8)) % 16)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp16, tmp22, tmp23)
    tmp25 = tmp14 + tmp24
    tmp26 = tmp2 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(rmask, tmp27, float("-inf"))
    tmp30 = triton_helpers.max2(tmp29, 1)[:, None]
    tmp31 = tmp26 - tmp30
    tmp32 = tl.exp(tmp31)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp32 / tmp36
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp37, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eg/cegkk6bxtzkd7ub3z2krcok32u6p7hn55boxc73muyxwibb46o4m.py
# Source Nodes: [reshape_38], Original ATen: [aten.clone]
# reshape_38 => clone_23
triton_poi_fused_clone_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = (xindex // 32768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (65536 + x0 + (98304*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bjgekoa6eayosbyeruoreurxyddv3on25eqdhsdgbik6dqgclx.py
# Source Nodes: [x_278, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_278 => add_88, mul_147, mul_148, sub_39
# x_281 => mul_149, sigmoid_37
triton_poi_fused__native_batch_norm_legit_no_training_silu_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 64
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*x1) + (8192*((x1 + (64*x0)) // 8192)) + (32768*x2) + (x0 % 128)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7zk7slkwu4zpzmd6tphaol3q4mj2hrfeojj2x2bzmbhdowretk.py
# Source Nodes: [x_283, x_289, x_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# x_283 => add_90, mul_151, mul_152, sub_40
# x_289 => add_91
# x_290 => mul_153, sigmoid_38
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1536
    y1 = (yindex // 1536)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (y0 + (1536*x2) + (98304*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + (y0 + (1536*x2) + (98304*y1)), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xo/cxohtjwmhkpsgbfx3rulxeuelx24zuwdjodamthtglnerjfxgylu.py
# Source Nodes: [x_292, x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_292 => add_93, mul_155, mul_156, sub_41
# x_297 => mul_157, sigmoid_39
# x_298 => mean_6
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_58', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 64.0
    tmp22 = tmp20 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1 = args
    args.clear()
    assert_size_stride(arg0_1, (24, ), (1, ))
    assert_size_stride(arg1_1, (24, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (63, 32), (32, 1))
    assert_size_stride(arg37_1, (63, 32), (32, 1))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (1024, ), (1, ))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (31, 64), (64, 1))
    assert_size_stride(arg59_1, (31, 64), (64, 1))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (1024, ), (1, ))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (31, 128), (128, 1))
    assert_size_stride(arg67_1, (31, 128), (128, 1))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (1536, ), (1, ))
    assert_size_stride(arg71_1, (1536, ), (1, ))
    assert_size_stride(arg72_1, (1536, ), (1, ))
    assert_size_stride(arg73_1, (1536, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (15, 128), (128, 1))
    assert_size_stride(arg77_1, (15, 128), (128, 1))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (1536, ), (1, ))
    assert_size_stride(arg81_1, (1536, ), (1, ))
    assert_size_stride(arg82_1, (1280, ), (1, ))
    assert_size_stride(arg83_1, (1280, ), (1, ))
    assert_size_stride(arg84_1, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg85_1, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(arg86_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg87_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg88_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg89_1, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg90_1, (8, ), (1, ))
    assert_size_stride(arg91_1, (64, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg92_1, (64, ), (1, ))
    assert_size_stride(arg93_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg94_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg95_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg96_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg97_1, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg98_1, (8, ), (1, ))
    assert_size_stride(arg99_1, (64, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg100_1, (64, ), (1, ))
    assert_size_stride(arg101_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg102_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg103_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg104_1, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg105_1, (8, ), (1, ))
    assert_size_stride(arg106_1, (128, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg109_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg110_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg111_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg112_1, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg113_1, (8, ), (1, ))
    assert_size_stride(arg114_1, (128, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg117_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg118_1, (384, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg119_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg120_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg121_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg122_1, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg123_1, (16, ), (1, ))
    assert_size_stride(arg124_1, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg127_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg128_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg129_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg130_1, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg131_1, (16, ), (1, ))
    assert_size_stride(arg132_1, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg135_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg136_1, (768, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg137_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg138_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg139_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg140_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg141_1, (1536, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg142_1, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg143_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg144_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg145_1, (1280, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg146_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg147_1, (1000, ), (1, ))
    assert_size_stride(arg148_1, (24, ), (1, ))
    assert_size_stride(arg149_1, (24, ), (1, ))
    assert_size_stride(arg150_1, (32, ), (1, ))
    assert_size_stride(arg151_1, (32, ), (1, ))
    assert_size_stride(arg152_1, (64, ), (1, ))
    assert_size_stride(arg153_1, (64, ), (1, ))
    assert_size_stride(arg154_1, (64, ), (1, ))
    assert_size_stride(arg155_1, (64, ), (1, ))
    assert_size_stride(arg156_1, (64, ), (1, ))
    assert_size_stride(arg157_1, (64, ), (1, ))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (256, ), (1, ))
    assert_size_stride(arg161_1, (256, ), (1, ))
    assert_size_stride(arg162_1, (64, ), (1, ))
    assert_size_stride(arg163_1, (64, ), (1, ))
    assert_size_stride(arg164_1, (64, ), (1, ))
    assert_size_stride(arg165_1, (64, ), (1, ))
    assert_size_stride(arg166_1, (256, ), (1, ))
    assert_size_stride(arg167_1, (256, ), (1, ))
    assert_size_stride(arg168_1, (128, ), (1, ))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (128, ), (1, ))
    assert_size_stride(arg171_1, (128, ), (1, ))
    assert_size_stride(arg172_1, (512, ), (1, ))
    assert_size_stride(arg173_1, (512, ), (1, ))
    assert_size_stride(arg174_1, (512, ), (1, ))
    assert_size_stride(arg175_1, (512, ), (1, ))
    assert_size_stride(arg176_1, (128, ), (1, ))
    assert_size_stride(arg177_1, (128, ), (1, ))
    assert_size_stride(arg178_1, (128, ), (1, ))
    assert_size_stride(arg179_1, (128, ), (1, ))
    assert_size_stride(arg180_1, (512, ), (1, ))
    assert_size_stride(arg181_1, (512, ), (1, ))
    assert_size_stride(arg182_1, (128, ), (1, ))
    assert_size_stride(arg183_1, (128, ), (1, ))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (128, ), (1, ))
    assert_size_stride(arg186_1, (512, ), (1, ))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (1024, ), (1, ))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (256, ), (1, ))
    assert_size_stride(arg197_1, (256, ), (1, ))
    assert_size_stride(arg198_1, (256, ), (1, ))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, ), (1, ))
    assert_size_stride(arg202_1, (256, ), (1, ))
    assert_size_stride(arg203_1, (256, ), (1, ))
    assert_size_stride(arg204_1, (256, ), (1, ))
    assert_size_stride(arg205_1, (256, ), (1, ))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, ), (1, ))
    assert_size_stride(arg208_1, (512, ), (1, ))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (512, ), (1, ))
    assert_size_stride(arg211_1, (512, ), (1, ))
    assert_size_stride(arg212_1, (1536, ), (1, ))
    assert_size_stride(arg213_1, (1536, ), (1, ))
    assert_size_stride(arg214_1, (1536, ), (1, ))
    assert_size_stride(arg215_1, (1536, ), (1, ))
    assert_size_stride(arg216_1, (512, ), (1, ))
    assert_size_stride(arg217_1, (512, ), (1, ))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (512, ), (1, ))
    assert_size_stride(arg220_1, (1536, ), (1, ))
    assert_size_stride(arg221_1, (1536, ), (1, ))
    assert_size_stride(arg222_1, (1280, ), (1, ))
    assert_size_stride(arg223_1, (1280, ), (1, ))
    assert_size_stride(arg224_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg224_1, buf0, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del arg224_1
        buf1 = empty_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg84_1, buf1, 72, 9, grid=grid(72, 9), stream=stream0)
        del arg84_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 24, 128, 128), (393216, 16384, 128, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_2.run(buf3, arg148_1, arg149_1, arg0_1, arg1_1, buf4, 192, 16384, grid=grid(192, 16384), stream=stream0)
        del arg0_1
        del arg148_1
        del arg149_1
        del arg1_1
        del buf3
        buf5 = empty_strided((32, 24, 3, 3), (216, 1, 72, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4, x_5], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_3.run(arg85_1, buf5, 768, 9, grid=grid(768, 9), stream=stream0)
        del arg85_1
        # Source Nodes: [x_4, x_5], Original ATen: [aten.convolution, aten.silu]
        buf6 = extern_kernels.convolution(buf4, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 32, 128, 128), (524288, 16384, 128, 1))
        del buf4
        del buf5
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_4.run(buf7, arg150_1, arg151_1, arg2_1, arg3_1, buf8, 256, 16384, grid=grid(256, 16384), stream=stream0)
        del arg150_1
        del arg151_1
        del arg2_1
        del arg3_1
        del buf7
        buf9 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10, x_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_5.run(arg86_1, buf9, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg86_1
        # Source Nodes: [x_10, x_9], Original ATen: [aten.convolution, aten.silu]
        buf10 = extern_kernels.convolution(buf8, buf9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del buf9
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_6.run(buf11, arg152_1, arg153_1, arg4_1, arg5_1, buf12, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del arg152_1
        del arg153_1
        del arg4_1
        del arg5_1
        # Source Nodes: [shortcut, x_16], Original ATen: [aten.convolution, aten.silu]
        buf13 = extern_kernels.convolution(buf12, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg87_1
        buf14 = buf13; del buf13  # reuse
        buf15 = reinterpret_tensor(buf11, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf11  # reuse
        # Source Nodes: [x_17, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_6.run(buf14, arg154_1, arg155_1, arg6_1, arg7_1, buf15, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del arg154_1
        del arg155_1
        del arg6_1
        del arg7_1
        del buf14
        buf16 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21, x_22], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_7.run(arg88_1, buf16, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg88_1
        # Source Nodes: [x_21, x_22], Original ATen: [aten.convolution, aten.silu]
        buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf18 = buf17; del buf17  # reuse
        buf19 = empty_strided((8, 64, 1, 1), (64, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf20 = reinterpret_tensor(buf19, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf19  # reuse
        # Source Nodes: [x_23, x_27, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8.run(buf18, buf20, arg156_1, arg157_1, arg8_1, arg9_1, 512, 4096, grid=grid(512), stream=stream0)
        del arg156_1
        del arg157_1
        del arg8_1
        del arg9_1
        # Source Nodes: [x_27, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf21 = extern_kernels.convolution(buf20, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg89_1
        del buf20
        buf22 = reinterpret_tensor(buf21, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf21  # reuse
        # Source Nodes: [x_27, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_9.run(buf22, arg90_1, 64, grid=grid(64), stream=stream0)
        del arg90_1
        # Source Nodes: [x_27, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        buf23 = extern_kernels.convolution(buf22, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg91_1
        del buf22
        buf24 = buf15; del buf15  # reuse
        # Source Nodes: [sigmoid, x_27, x_29, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_10.run(buf18, buf23, arg92_1, buf24, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del arg92_1
        del buf18
        # Source Nodes: [sigmoid, x_27, x_29, x_30, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        buf25 = extern_kernels.convolution(buf24, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg93_1
        del buf24
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf12, arg94_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg94_1
        buf27 = buf25; del buf25  # reuse
        buf28 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_1, x_31, x_39, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_11.run(buf27, arg158_1, arg159_1, arg10_1, arg11_1, buf26, arg160_1, arg161_1, arg12_1, arg13_1, buf28, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg13_1
        del arg158_1
        del arg159_1
        del arg160_1
        del arg161_1
        del buf26
        # Source Nodes: [shortcut_1, x_44], Original ATen: [aten.convolution, aten.silu]
        buf29 = extern_kernels.convolution(buf28, arg95_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg95_1
        buf30 = buf29; del buf29  # reuse
        buf31 = buf12; del buf12  # reuse
        # Source Nodes: [x_45, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_6.run(buf30, arg162_1, arg163_1, arg14_1, arg15_1, buf31, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del arg14_1
        del arg15_1
        del arg162_1
        del arg163_1
        del buf30
        buf32 = buf16; del buf16  # reuse
        # Source Nodes: [x_49, x_50], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_7.run(arg96_1, buf32, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg96_1
        # Source Nodes: [x_49, x_50], Original ATen: [aten.convolution, aten.silu]
        buf33 = extern_kernels.convolution(buf31, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del buf32
        buf34 = buf33; del buf33  # reuse
        buf35 = reinterpret_tensor(buf23, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf23  # reuse
        buf36 = reinterpret_tensor(buf35, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf35  # reuse
        # Source Nodes: [x_51, x_55, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8.run(buf34, buf36, arg164_1, arg165_1, arg16_1, arg17_1, 512, 4096, grid=grid(512), stream=stream0)
        del arg164_1
        del arg165_1
        del arg16_1
        del arg17_1
        # Source Nodes: [x_55, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf37 = extern_kernels.convolution(buf36, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg97_1
        del buf36
        buf38 = reinterpret_tensor(buf37, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf37  # reuse
        # Source Nodes: [x_55, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_9.run(buf38, arg98_1, 64, grid=grid(64), stream=stream0)
        del arg98_1
        # Source Nodes: [x_55, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        buf39 = extern_kernels.convolution(buf38, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg99_1
        del buf38
        buf40 = buf31; del buf31  # reuse
        # Source Nodes: [sigmoid_1, x_55, x_57, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_10.run(buf34, buf39, arg100_1, buf40, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del arg100_1
        del buf34
        del buf39
        # Source Nodes: [sigmoid_1, x_55, x_57, x_58, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        buf41 = extern_kernels.convolution(buf40, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg101_1
        buf42 = buf41; del buf41  # reuse
        buf43 = reinterpret_tensor(buf27, (8, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf27  # reuse
        # Source Nodes: [shortcut_2, x_59, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_12.run(buf42, arg166_1, arg167_1, arg18_1, arg19_1, buf28, buf43, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        del arg166_1
        del arg167_1
        del arg18_1
        del arg19_1
        del buf28
        del buf42
        # Source Nodes: [shortcut_2, x_67], Original ATen: [aten.convolution, aten.silu]
        buf44 = extern_kernels.convolution(buf43, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg102_1
        buf45 = buf44; del buf44  # reuse
        buf46 = reinterpret_tensor(buf8, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf8  # reuse
        # Source Nodes: [x_68, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_13.run(buf45, arg168_1, arg169_1, arg20_1, arg21_1, buf46, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del arg168_1
        del arg169_1
        del arg20_1
        del arg21_1
        del buf45
        buf47 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72, x_73], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_14.run(arg103_1, buf47, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg103_1
        # Source Nodes: [x_72, x_73], Original ATen: [aten.convolution, aten.silu]
        buf48 = extern_kernels.convolution(buf46, buf47, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf49 = buf48; del buf48  # reuse
        buf50 = empty_strided((8, 128, 1, 1), (128, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf51 = reinterpret_tensor(buf50, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf50  # reuse
        # Source Nodes: [x_74, x_78, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_15.run(buf49, buf51, arg170_1, arg171_1, arg22_1, arg23_1, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg170_1
        del arg171_1
        del arg22_1
        del arg23_1
        # Source Nodes: [x_78, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf52 = extern_kernels.convolution(buf51, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg104_1
        del buf51
        buf53 = reinterpret_tensor(buf52, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf52  # reuse
        # Source Nodes: [x_78, x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_9.run(buf53, arg105_1, 64, grid=grid(64), stream=stream0)
        del arg105_1
        # Source Nodes: [x_78, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        buf54 = extern_kernels.convolution(buf53, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg106_1
        del buf53
        buf55 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2, x_78, x_80, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_16.run(buf49, buf54, arg107_1, buf55, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg107_1
        del buf49
        # Source Nodes: [sigmoid_2, x_78, x_80, x_81, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        buf56 = extern_kernels.convolution(buf55, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg108_1
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf43, arg109_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg109_1
        del buf43
        buf58 = buf56; del buf56  # reuse
        buf59 = reinterpret_tensor(buf46, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf46  # reuse
        # Source Nodes: [shortcut_3, x_82, x_90, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_17.run(buf58, arg172_1, arg173_1, arg24_1, arg25_1, buf57, arg174_1, arg175_1, arg26_1, arg27_1, buf59, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del arg24_1
        del arg25_1
        del arg26_1
        del arg27_1
        del buf57
        # Source Nodes: [shortcut_3, x_95], Original ATen: [aten.convolution, aten.silu]
        buf60 = extern_kernels.convolution(buf59, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg110_1
        buf61 = buf60; del buf60  # reuse
        buf62 = buf55; del buf55  # reuse
        # Source Nodes: [x_100, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_18.run(buf61, arg176_1, arg177_1, arg28_1, arg29_1, buf62, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg176_1
        del arg177_1
        del arg28_1
        del arg29_1
        del buf61
        buf63 = buf47; del buf47  # reuse
        # Source Nodes: [x_100, x_101], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_14.run(arg111_1, buf63, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg111_1
        # Source Nodes: [x_100, x_101], Original ATen: [aten.convolution, aten.silu]
        buf64 = extern_kernels.convolution(buf62, buf63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del buf63
        buf65 = buf64; del buf64  # reuse
        buf66 = reinterpret_tensor(buf54, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf54  # reuse
        buf67 = reinterpret_tensor(buf66, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf66  # reuse
        # Source Nodes: [x_102, x_106, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_15.run(buf65, buf67, arg178_1, arg179_1, arg30_1, arg31_1, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg178_1
        del arg179_1
        del arg30_1
        del arg31_1
        # Source Nodes: [x_106, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf68 = extern_kernels.convolution(buf67, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg112_1
        del buf67
        buf69 = reinterpret_tensor(buf68, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf68  # reuse
        # Source Nodes: [x_106, x_se_12, x_se_13, x_se_14], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_9.run(buf69, arg113_1, 64, grid=grid(64), stream=stream0)
        del arg113_1
        # Source Nodes: [x_106, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        buf70 = extern_kernels.convolution(buf69, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg114_1
        del buf69
        buf71 = buf62; del buf62  # reuse
        # Source Nodes: [sigmoid_3, x_106, x_108, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_16.run(buf65, buf70, arg115_1, buf71, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg115_1
        del buf65
        del buf70
        # Source Nodes: [sigmoid_3, x_106, x_108, x_109, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        buf72 = extern_kernels.convolution(buf71, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg116_1
        buf73 = buf72; del buf72  # reuse
        buf74 = reinterpret_tensor(buf58, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf58  # reuse
        # Source Nodes: [shortcut_4, x_110, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_19.run(buf73, arg180_1, arg181_1, arg32_1, arg33_1, buf59, buf74, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        del arg180_1
        del arg181_1
        del arg32_1
        del arg33_1
        del buf59
        # Source Nodes: [shortcut_4, x_118], Original ATen: [aten.convolution, aten.silu]
        buf75 = extern_kernels.convolution(buf74, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg117_1
        buf76 = buf75; del buf75  # reuse
        buf77 = buf71; del buf71  # reuse
        # Source Nodes: [x_119, x_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_18.run(buf76, arg182_1, arg183_1, arg34_1, arg35_1, buf77, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg182_1
        del arg183_1
        del arg34_1
        del arg35_1
        # Source Nodes: [x_123, x_125], Original ATen: [aten.convolution, aten.silu]
        buf78 = extern_kernels.convolution(buf77, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 384, 32, 32), (393216, 1024, 32, 1))
        del arg118_1
        buf79 = reinterpret_tensor(buf77, (8, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf77  # reuse
        # Source Nodes: [reshape], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf78, buf79, 1048576, grid=grid(1048576), stream=stream0)
        buf80 = buf76; del buf76  # reuse
        # Source Nodes: [k_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf78, buf80, 1048576, grid=grid(1048576), stream=stream0)
        buf81 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (32, 1024, 32), (32768, 1, 1024), 0), reinterpret_tensor(buf80, (32, 32, 1024), (32768, 1024, 1), 0), out=buf81)
        buf82 = reinterpret_tensor(buf80, (32, 32, 32, 32), (32768, 1024, 32, 1), 0); del buf80  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf79, buf82, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf83 = empty((32768, 63), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (32768, 32), (32, 1), 0), reinterpret_tensor(arg37_1, (32, 63), (1, 32), 0), out=buf83)
        del arg37_1
        buf84 = buf82; del buf82  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf79, buf84, 32768, 32, grid=grid(32768, 32), stream=stream0)
        buf85 = empty((32768, 63), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (32768, 32), (32, 1), 0), reinterpret_tensor(arg36_1, (32, 63), (1, 32), 0), out=buf85)
        del arg36_1
        buf88 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1, mul_4], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_24.run(buf81, buf83, buf85, buf88, 32768, 1024, grid=grid(32768), stream=stream0)
        del buf81
        del buf83
        del buf85
        buf89 = reinterpret_tensor(buf84, (8, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf84  # reuse
        # Source Nodes: [reshape_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf78, buf89, 1048576, grid=grid(1048576), stream=stream0)
        del buf78
        buf90 = reinterpret_tensor(buf79, (32, 1024, 32), (32768, 32, 1), 0); del buf79  # reuse
        # Source Nodes: [attn, attn_1, matmul_3, mul_4], Original ATen: [aten._softmax, aten.add, aten.bmm, aten.mul]
        extern_kernels.bmm(buf88, reinterpret_tensor(buf89, (32, 1024, 32), (32768, 1, 1024), 0), out=buf90)
        del buf88
        buf91 = reinterpret_tensor(buf89, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf89  # reuse
        buf92 = buf91; del buf91  # reuse
        # Source Nodes: [x_135, x_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_26.run(buf92, buf90, arg184_1, arg185_1, arg38_1, arg39_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg184_1
        del arg185_1
        del arg38_1
        del arg39_1
        del buf90
        # Source Nodes: [x_138, x_139], Original ATen: [aten.convolution, aten.silu]
        buf93 = extern_kernels.convolution(buf92, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg119_1
        buf94 = buf93; del buf93  # reuse
        buf95 = reinterpret_tensor(buf73, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf73  # reuse
        # Source Nodes: [shortcut_5, x_140, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_19.run(buf94, arg186_1, arg187_1, arg40_1, arg41_1, buf74, buf95, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        del arg186_1
        del arg187_1
        del arg40_1
        del arg41_1
        del buf74
        del buf94
        # Source Nodes: [shortcut_5, x_147], Original ATen: [aten.convolution, aten.silu]
        buf96 = extern_kernels.convolution(buf95, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg120_1
        buf97 = buf96; del buf96  # reuse
        buf98 = reinterpret_tensor(buf40, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf40  # reuse
        # Source Nodes: [x_148, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_27.run(buf97, arg188_1, arg189_1, arg42_1, arg43_1, buf98, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        del arg188_1
        del arg189_1
        del arg42_1
        del arg43_1
        del buf97
        buf99 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_152, x_153], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_28.run(arg121_1, buf99, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg121_1
        # Source Nodes: [x_152, x_153], Original ATen: [aten.convolution, aten.silu]
        buf100 = extern_kernels.convolution(buf98, buf99, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf101 = buf100; del buf100  # reuse
        buf102 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf103 = reinterpret_tensor(buf102, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf102  # reuse
        # Source Nodes: [x_154, x_158, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf101, buf103, arg190_1, arg191_1, arg44_1, arg45_1, 2048, 256, grid=grid(2048), stream=stream0)
        del arg190_1
        del arg191_1
        del arg44_1
        del arg45_1
        # Source Nodes: [x_158, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf104 = extern_kernels.convolution(buf103, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg122_1
        del buf103
        buf105 = reinterpret_tensor(buf104, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf104  # reuse
        # Source Nodes: [x_158, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_30.run(buf105, arg123_1, 128, grid=grid(128), stream=stream0)
        del arg123_1
        # Source Nodes: [x_158, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        buf106 = extern_kernels.convolution(buf105, arg124_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg124_1
        del buf105
        buf107 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4, x_158, x_160, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_31.run(buf101, buf106, arg125_1, buf107, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg125_1
        del buf101
        # Source Nodes: [sigmoid_4, x_158, x_160, x_161, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        buf108 = extern_kernels.convolution(buf107, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg126_1
        # Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf95, arg127_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg127_1
        del buf95
        buf110 = buf108; del buf108  # reuse
        buf111 = reinterpret_tensor(buf98, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf98  # reuse
        # Source Nodes: [shortcut_6, x_162, x_170, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_32.run(buf110, arg192_1, arg193_1, arg46_1, arg47_1, buf109, arg194_1, arg195_1, arg48_1, arg49_1, buf111, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        del arg46_1
        del arg47_1
        del arg48_1
        del arg49_1
        del buf109
        # Source Nodes: [shortcut_6, x_175], Original ATen: [aten.convolution, aten.silu]
        buf112 = extern_kernels.convolution(buf111, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg128_1
        buf113 = buf112; del buf112  # reuse
        buf114 = buf107; del buf107  # reuse
        # Source Nodes: [x_176, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_33.run(buf113, arg196_1, arg197_1, arg50_1, arg51_1, buf114, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg196_1
        del arg197_1
        del arg50_1
        del arg51_1
        del buf113
        buf115 = buf99; del buf99  # reuse
        # Source Nodes: [x_180, x_181], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_28.run(arg129_1, buf115, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg129_1
        # Source Nodes: [x_180, x_181], Original ATen: [aten.convolution, aten.silu]
        buf116 = extern_kernels.convolution(buf114, buf115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 256, 16, 16), (65536, 256, 16, 1))
        del buf115
        buf117 = buf116; del buf116  # reuse
        buf118 = reinterpret_tensor(buf106, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf106  # reuse
        buf119 = reinterpret_tensor(buf118, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf118  # reuse
        # Source Nodes: [x_182, x_186, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf117, buf119, arg198_1, arg199_1, arg52_1, arg53_1, 2048, 256, grid=grid(2048), stream=stream0)
        del arg198_1
        del arg199_1
        del arg52_1
        del arg53_1
        # Source Nodes: [x_186, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf120 = extern_kernels.convolution(buf119, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg130_1
        del buf119
        buf121 = reinterpret_tensor(buf120, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf120  # reuse
        # Source Nodes: [x_186, x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_30.run(buf121, arg131_1, 128, grid=grid(128), stream=stream0)
        del arg131_1
        # Source Nodes: [x_186, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        buf122 = extern_kernels.convolution(buf121, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg132_1
        del buf121
        buf123 = buf114; del buf114  # reuse
        # Source Nodes: [sigmoid_5, x_186, x_188, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_31.run(buf117, buf122, arg133_1, buf123, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg133_1
        del buf117
        del buf122
        # Source Nodes: [sigmoid_5, x_186, x_188, x_189, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        buf124 = extern_kernels.convolution(buf123, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg134_1
        buf125 = buf124; del buf124  # reuse
        buf126 = reinterpret_tensor(buf110, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf110  # reuse
        # Source Nodes: [shortcut_7, x_190, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_34.run(buf125, arg200_1, arg201_1, arg54_1, arg55_1, buf111, buf126, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del arg200_1
        del arg201_1
        del arg54_1
        del arg55_1
        # Source Nodes: [shortcut_7, x_198], Original ATen: [aten.convolution, aten.silu]
        buf127 = extern_kernels.convolution(buf126, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg135_1
        buf128 = buf127; del buf127  # reuse
        buf129 = buf123; del buf123  # reuse
        # Source Nodes: [x_199, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_33.run(buf128, arg202_1, arg203_1, arg56_1, arg57_1, buf129, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg202_1
        del arg203_1
        del arg56_1
        del arg57_1
        # Source Nodes: [x_203, x_205], Original ATen: [aten.convolution, aten.silu]
        buf130 = extern_kernels.convolution(buf129, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 768, 16, 16), (196608, 256, 16, 1))
        del arg136_1
        buf131 = reinterpret_tensor(buf129, (8, 256, 16, 16), (65536, 256, 16, 1), 0); del buf129  # reuse
        # Source Nodes: [reshape_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf130, buf131, 524288, grid=grid(524288), stream=stream0)
        buf132 = buf128; del buf128  # reuse
        # Source Nodes: [k_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_36.run(buf130, buf132, 524288, grid=grid(524288), stream=stream0)
        buf133 = reinterpret_tensor(buf125, (32, 256, 256), (65536, 256, 1), 0); del buf125  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf131, (32, 256, 64), (16384, 1, 256), 0), reinterpret_tensor(buf132, (32, 64, 256), (16384, 256, 1), 0), out=buf133)
        buf134 = reinterpret_tensor(buf132, (32, 16, 16, 64), (16384, 1024, 64, 1), 0); del buf132  # reuse
        # Source Nodes: [x_210], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf131, buf134, 512, 1024, grid=grid(512, 1024), stream=stream0)
        buf135 = empty((8192, 31), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (8192, 64), (64, 1), 0), reinterpret_tensor(arg59_1, (64, 31), (1, 64), 0), out=buf135)
        del arg59_1
        buf136 = buf134; del buf134  # reuse
        # Source Nodes: [x_206], Original ATen: [aten.clone]
        triton_poi_fused_clone_38.run(buf131, buf136, 8192, 64, grid=grid(8192, 64), stream=stream0)
        buf137 = empty((8192, 31), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (8192, 64), (64, 1), 0), reinterpret_tensor(arg58_1, (64, 31), (1, 64), 0), out=buf137)
        del arg58_1
        buf140 = reinterpret_tensor(buf111, (32, 256, 256), (65536, 256, 1), 0); del buf111  # reuse
        # Source Nodes: [attn_2, attn_3, mul_7], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_39.run(buf133, buf135, buf137, buf140, 8192, 256, grid=grid(8192), stream=stream0)
        del buf133
        buf141 = reinterpret_tensor(buf136, (8, 256, 16, 16), (65536, 256, 16, 1), 0); del buf136  # reuse
        # Source Nodes: [reshape_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf130, buf141, 524288, grid=grid(524288), stream=stream0)
        del buf130
        buf142 = reinterpret_tensor(buf131, (32, 256, 64), (16384, 64, 1), 0); del buf131  # reuse
        # Source Nodes: [attn_2, attn_3, matmul_7, mul_7], Original ATen: [aten._softmax, aten.add, aten.bmm, aten.mul]
        extern_kernels.bmm(buf140, reinterpret_tensor(buf141, (32, 256, 64), (16384, 1, 256), 0), out=buf142)
        buf143 = reinterpret_tensor(buf141, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf141  # reuse
        buf144 = buf143; del buf143  # reuse
        # Source Nodes: [x_215, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_41.run(buf144, buf142, arg204_1, arg205_1, arg60_1, arg61_1, 524288, grid=grid(524288), stream=stream0)
        del arg204_1
        del arg205_1
        del arg60_1
        del arg61_1
        del buf142
        # Source Nodes: [x_218, x_219], Original ATen: [aten.convolution, aten.silu]
        buf145 = extern_kernels.convolution(buf144, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg137_1
        del buf144
        buf146 = buf145; del buf145  # reuse
        buf147 = reinterpret_tensor(buf140, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf140  # reuse
        # Source Nodes: [shortcut_8, x_220, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_34.run(buf146, arg206_1, arg207_1, arg62_1, arg63_1, buf126, buf147, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del arg206_1
        del arg207_1
        del arg62_1
        del arg63_1
        # Source Nodes: [shortcut_8, x_227], Original ATen: [aten.convolution, aten.silu]
        buf148 = extern_kernels.convolution(buf147, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg138_1
        buf149 = buf148; del buf148  # reuse
        buf150 = reinterpret_tensor(buf92, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf92  # reuse
        # Source Nodes: [x_228, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_42.run(buf149, arg208_1, arg209_1, arg64_1, arg65_1, buf150, 4096, 256, grid=grid(4096, 256), stream=stream0)
        del arg208_1
        del arg209_1
        del arg64_1
        del arg65_1
        # Source Nodes: [x_232, x_234], Original ATen: [aten.convolution, aten.silu]
        buf151 = extern_kernels.convolution(buf150, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del arg139_1
        buf152 = reinterpret_tensor(buf150, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf150  # reuse
        # Source Nodes: [reshape_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf151, buf152, 1048576, grid=grid(1048576), stream=stream0)
        buf153 = buf149; del buf149  # reuse
        # Source Nodes: [k_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf151, buf153, 1048576, grid=grid(1048576), stream=stream0)
        buf154 = reinterpret_tensor(buf146, (32, 256, 256), (65536, 256, 1), 0); del buf146  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf152, (32, 256, 128), (32768, 1, 256), 0), reinterpret_tensor(buf153, (32, 128, 256), (32768, 256, 1), 0), out=buf154)
        buf155 = reinterpret_tensor(buf153, (32, 16, 16, 128), (32768, 2048, 128, 1), 0); del buf153  # reuse
        # Source Nodes: [x_239], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf152, buf155, 512, 2048, grid=grid(512, 2048), stream=stream0)
        buf156 = buf137; del buf137  # reuse
        # Source Nodes: [x_239], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (8192, 128), (128, 1), 0), reinterpret_tensor(arg67_1, (128, 31), (1, 128), 0), out=buf156)
        del arg67_1
        buf157 = buf155; del buf155  # reuse
        # Source Nodes: [x_235], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf152, buf157, 8192, 128, grid=grid(8192, 128), stream=stream0)
        buf158 = buf135; del buf135  # reuse
        # Source Nodes: [x_235], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (8192, 128), (128, 1), 0), reinterpret_tensor(arg66_1, (128, 31), (1, 128), 0), out=buf158)
        del arg66_1
        buf161 = reinterpret_tensor(buf126, (32, 256, 256), (65536, 256, 1), 0); del buf126  # reuse
        # Source Nodes: [attn_4, attn_5, mul_8], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_45.run(buf154, buf156, buf158, buf161, 8192, 256, grid=grid(8192), stream=stream0)
        del buf154
        del buf156
        del buf158
        buf162 = reinterpret_tensor(buf157, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf157  # reuse
        # Source Nodes: [reshape_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf151, buf162, 1048576, grid=grid(1048576), stream=stream0)
        del buf151
        buf163 = reinterpret_tensor(buf152, (32, 256, 128), (32768, 128, 1), 0); del buf152  # reuse
        # Source Nodes: [attn_4, attn_5, matmul_11, mul_8], Original ATen: [aten._softmax, aten.add, aten.bmm, aten.mul]
        extern_kernels.bmm(buf161, reinterpret_tensor(buf162, (32, 256, 128), (32768, 1, 256), 0), out=buf163)
        del buf161
        del buf162
        buf164 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d]
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_46.run(buf163, arg210_1, arg211_1, arg68_1, arg69_1, buf164, 262144, grid=grid(262144), stream=stream0)
        del arg210_1
        del arg211_1
        del arg68_1
        del arg69_1
        del buf163
        buf165 = buf164; del buf164  # reuse
        # Source Nodes: [x_247], Original ATen: [aten.silu]
        triton_poi_fused_silu_47.run(buf165, 4096, 64, grid=grid(4096, 64), stream=stream0)
        # Source Nodes: [x_247, x_248], Original ATen: [aten.convolution, aten.silu]
        buf166 = extern_kernels.convolution(buf165, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del arg140_1
        # Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf147, arg141_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del arg141_1
        del buf147
        buf168 = buf166; del buf166  # reuse
        buf169 = empty_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_9, x_249, x_256, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_48.run(buf168, arg212_1, arg213_1, arg70_1, arg71_1, buf167, arg214_1, arg215_1, arg72_1, arg73_1, buf169, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del arg70_1
        del arg71_1
        del arg72_1
        del arg73_1
        del buf167
        del buf168
        # Source Nodes: [shortcut_9, x_261], Original ATen: [aten.convolution, aten.silu]
        buf170 = extern_kernels.convolution(buf169, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg142_1
        buf171 = buf170; del buf170  # reuse
        buf172 = buf165; del buf165  # reuse
        # Source Nodes: [x_262, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_49.run(buf171, arg216_1, arg217_1, arg74_1, arg75_1, buf172, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg216_1
        del arg217_1
        del arg74_1
        del arg75_1
        # Source Nodes: [x_266, x_268], Original ATen: [aten.convolution, aten.silu]
        buf173 = extern_kernels.convolution(buf172, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del arg143_1
        buf174 = reinterpret_tensor(buf172, (8, 512, 8, 8), (32768, 64, 8, 1), 0); del buf172  # reuse
        # Source Nodes: [reshape_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_50.run(buf173, buf174, 262144, grid=grid(262144), stream=stream0)
        buf175 = buf171; del buf171  # reuse
        # Source Nodes: [k_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(buf173, buf175, 262144, grid=grid(262144), stream=stream0)
        buf176 = empty((32, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf174, (32, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf175, (32, 128, 64), (8192, 64, 1), 0), out=buf176)
        buf177 = reinterpret_tensor(buf175, (32, 8, 8, 128), (8192, 1024, 128, 1), 0); del buf175  # reuse
        # Source Nodes: [x_273], Original ATen: [aten.clone]
        triton_poi_fused_clone_52.run(buf174, buf177, 256, 1024, grid=grid(256, 1024), stream=stream0)
        buf178 = empty((2048, 15), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_273], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (2048, 128), (128, 1), 0), reinterpret_tensor(arg77_1, (128, 15), (1, 128), 0), out=buf178)
        del arg77_1
        buf179 = buf177; del buf177  # reuse
        # Source Nodes: [x_269], Original ATen: [aten.clone]
        triton_poi_fused_clone_53.run(buf174, buf179, 2048, 128, grid=grid(2048, 128), stream=stream0)
        buf180 = empty((2048, 15), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (2048, 128), (128, 1), 0), reinterpret_tensor(arg76_1, (128, 15), (1, 128), 0), out=buf180)
        del arg76_1
        buf183 = empty((32, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_6, attn_7, mul_9], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_54.run(buf176, buf178, buf180, buf183, 2048, 64, grid=grid(2048), stream=stream0)
        del buf176
        del buf178
        del buf180
        buf184 = reinterpret_tensor(buf179, (8, 512, 8, 8), (32768, 64, 8, 1), 0); del buf179  # reuse
        # Source Nodes: [reshape_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_55.run(buf173, buf184, 262144, grid=grid(262144), stream=stream0)
        buf185 = reinterpret_tensor(buf174, (32, 64, 128), (8192, 128, 1), 0); del buf174  # reuse
        # Source Nodes: [attn_6, attn_7, matmul_15, mul_9], Original ATen: [aten._softmax, aten.add, aten.bmm, aten.mul]
        extern_kernels.bmm(buf183, reinterpret_tensor(buf184, (32, 64, 128), (8192, 1, 64), 0), out=buf185)
        del buf183
        buf186 = reinterpret_tensor(buf184, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf184  # reuse
        buf187 = buf186; del buf186  # reuse
        # Source Nodes: [x_278, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_56.run(buf187, buf185, arg218_1, arg219_1, arg78_1, arg79_1, 262144, grid=grid(262144), stream=stream0)
        del arg218_1
        del arg219_1
        del arg78_1
        del arg79_1
        del buf185
        # Source Nodes: [x_281, x_282], Original ATen: [aten.convolution, aten.silu]
        buf188 = extern_kernels.convolution(buf187, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del arg144_1
        del buf187
        buf189 = buf188; del buf188  # reuse
        buf190 = reinterpret_tensor(buf173, (8, 1536, 8, 8), (98304, 1, 12288, 1536), 0); del buf173  # reuse
        # Source Nodes: [x_283, x_289, x_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_57.run(buf189, arg220_1, arg221_1, arg80_1, arg81_1, buf169, buf190, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del arg220_1
        del arg221_1
        del arg80_1
        del arg81_1
        del buf169
        del buf189
        # Source Nodes: [x_290, x_291], Original ATen: [aten.convolution, aten.silu]
        buf191 = extern_kernels.convolution(buf190, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 1280, 8, 8), (81920, 64, 8, 1))
        del arg145_1
        del buf190
        buf192 = buf191; del buf191  # reuse
        buf193 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf194 = reinterpret_tensor(buf193, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf193  # reuse
        # Source Nodes: [x_292, x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_58.run(buf192, buf194, arg222_1, arg223_1, arg82_1, arg83_1, 10240, 64, grid=grid(10240), stream=stream0)
        del arg222_1
        del arg223_1
        del arg82_1
        del arg83_1
        del buf192
        buf195 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_302], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg147_1, reinterpret_tensor(buf194, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg146_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf195)
        del arg146_1
        del arg147_1
        return (buf195, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((63, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((63, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((31, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((31, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((31, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((31, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((15, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((15, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((64, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((64, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1536, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1280, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('sebotnet33ts_256', benchmark_compiled_module)
