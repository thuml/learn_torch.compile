
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3mdeneejappqqe5fk6zuhypf6spx7mpv4obazedrrc25iia6ca.py
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
    xnumel = 50176
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
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2e3bebqzvc7zi4dei55h5bgpn26w7rkjmbs2qkuumsyjqtkjtk.py
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
    size_hints=[256, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (147*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c2/cc24e62426mxvavozylrfg2vi3xbig7xyrp3lc6v4e5y4zxhfaf2.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c256efoixkmaq5hdhm62dhptk6dl6abe4qzid5cv2jq2urrnfjw4.py
# Source Nodes: [shortcut, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
# shortcut => max_pool2d_with_indices
# x_1 => add_1, mul_1, mul_2, sub
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 56)
    x2 = xindex % 56
    y4 = yindex
    x5 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = (-1) + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x2)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + (2*x2) + (224*x3) + (12544*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-112) + (2*x2) + (224*x3) + (12544*y4)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-111) + (2*x2) + (224*x3) + (12544*y4)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x2) + (224*x3) + (12544*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x2) + (224*x3) + (12544*y4)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x2) + (224*x3) + (12544*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (111 + (2*x2) + (224*x3) + (12544*y4)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (112 + (2*x2) + (224*x3) + (12544*y4)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (113 + (2*x2) + (224*x3) + (12544*y4)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tl.store(out_ptr0 + (y0 + (64*x5) + (200704*y1)), tmp69, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kr/ckru2fcdch43xfwxforpfmmld7sqbedunfoetq7khoffr5svb5qw.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_1 => add_3, mul_4, mul_5, sub_1
# out_2 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2609152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 104
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tp/ctpm5gdqz3gl5n324vpot6mezjszyeu4dev24c7arcffq4gkvu5d.py
# Source Nodes: [sp_1], Original ATen: [aten.convolution]
# sp_1 => convolution_2
triton_poi_fused_convolution_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (26*x2) + (81536*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3p4hodv76hhkaeaeow6pucv3rtaa6u62s5n2p36aci7svexmz4z.py
# Source Nodes: [sp_1], Original ATen: [aten.convolution]
# sp_1 => convolution_2
triton_poi_fused_convolution_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 676
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (26*x2) + (234*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ik/cik62avpqr63oipwqrjiz6i47o6ovv4d3qxlrrou3erv3busabl7.py
# Source Nodes: [sp_5], Original ATen: [aten.convolution]
# sp_5 => convolution_3
triton_poi_fused_convolution_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (81536 + x2 + (3136*y0) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (26*x2) + (81536*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggjvipw6e7ckhmmholfjhidkys4bc4ibftdd7vit7aov476442p.py
# Source Nodes: [sp_9], Original ATen: [aten.convolution]
# sp_9 => convolution_4
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (163072 + x2 + (3136*y0) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (26*x2) + (81536*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqwezx3hp7xqlx7hcraukgfi5jmtqg652iskhvriko2vkxbcazrn.py
# Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer1___0___pool => avg_pool2d
triton_poi_fused_avg_pool2d_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 81536)
    x6 = xindex % 81536
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (244551 + x6 + (326144*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (244552 + x6 + (326144*x3)), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (244553 + x6 + (326144*x3)), tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (244607 + x6 + (326144*x3)), tmp36 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (244608 + x6 + (326144*x3)), tmp41 & xmask, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (244609 + x6 + (326144*x3)), tmp46 & xmask, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (244663 + x6 + (326144*x3)), tmp55 & xmask, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (244664 + x6 + (326144*x3)), tmp60 & xmask, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (244665 + x6 + (326144*x3)), tmp65 & xmask, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 57, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6 + (326144*x3)), tmp145, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c755nvj6nqskraev244ib4bkrc36dn2345iuenjswysoarg4rs7q.py
# Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_2 => add_5, mul_7, mul_8, sub_2
# sp_3 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 26
    x2 = (xindex // 81536)
    x4 = xindex % 81536
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x4 + (326144*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6eljcwzxwwxlwuqjhtweb76x6ho3n7yo7tlty2niztakv34pypd.py
# Source Nodes: [out_4], Original ATen: [aten.convolution]
# out_4 => convolution_5
triton_poi_fused_convolution_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (326144*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/cztps75pnpdtzpqbkeq3lmcsryo6eeowptrzidxwtfn6ifbkzqx6.py
# Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_5 => add_11, mul_16, mul_17, sub_5
# out_6 => add_14
# shortcut_1 => add_13, mul_19, mul_20, sub_6
# shortcut_2 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 3136
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
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
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (256*x2) + (802816*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c755hpnhx3iaikebghp35jaa3avetwd5rvravqitjdkqheyndf2x.py
# Source Nodes: [sp_15, sp_16, sp_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_15 => add_18, mul_25, mul_26, sub_8
# sp_16 => relu_7
# sp_17 => add_19
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (81536 + x2 + (3136*y0) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (3136*y0) + (326144*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (26*x2) + (81536*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqfbaz7h4n2woddp3jtjimkmamtf4t4ompvdjqnvu5zryrgrfhjo.py
# Source Nodes: [sp_19, sp_20, sp_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_19 => add_21, mul_28, mul_29, sub_9
# sp_20 => relu_8
# sp_21 => add_22
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (163072 + x2 + (3136*y0) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (3136*y0) + (326144*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (26*x2) + (81536*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ry/cry3z6vovek7yrbtwj4qhhktjjd7kms6wzut3iwij7s35dirbng4.py
# Source Nodes: [cat_64], Original ATen: [aten.cat]
# cat_64 => cat_1
triton_poi_fused_cat_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 81536
    x1 = (xindex // 81536)
    tmp0 = tl.load(in_ptr0 + (244608 + x0 + (326144*x1)), xmask)
    tl.store(out_ptr0 + (x0 + (326144*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygacnnlaexn2ip4uxip2e6nqwux22fm2faijjughpdgtt4wng6i.py
# Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_13 => add_26, mul_34, mul_35, sub_11
# out_14 => add_27
# shortcut_3 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2k/c2kb2iehixz75klual7jqzl2dfhknfbz2xopvghfvh564u2kx377.py
# Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_25 => add_42, mul_52, mul_53, sub_17
# out_26 => relu_16
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5218304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 208
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/te/ctez7tuntd6deynevqpkpdkshys6vm3p636csvhtaxw7aq6xuixb.py
# Source Nodes: [sp_40], Original ATen: [aten.convolution]
# sp_40 => convolution_18
triton_poi_fused_convolution_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (652288*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (52*x2) + (163072*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xe/cxeltelivpaxufdscko7zfxwcgprvnqir3m5pa3m3m2imvc2if6e.py
# Source Nodes: [sp_40], Original ATen: [aten.convolution]
# sp_40 => convolution_18
triton_poi_fused_convolution_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2704
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (52*x2) + (468*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbncnagz6wzqllsv5pie652bgvec2bppmkdot5hzq65daivrfev7.py
# Source Nodes: [sp_44], Original ATen: [aten.convolution]
# sp_44 => convolution_19
triton_poi_fused_convolution_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (163072 + x2 + (3136*y0) + (652288*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (52*x2) + (163072*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/32/c32nsbztx2tcctqqytsvzjh26ionexkvgjpvnrukw4vjnzf6yuno.py
# Source Nodes: [sp_48], Original ATen: [aten.convolution]
# sp_48 => convolution_20
triton_poi_fused_convolution_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (326144 + x2 + (3136*y0) + (652288*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (52*x2) + (163072*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w3/cw3unsqlfn5ii4pp67pt36rcaq3uisdjteutar323hdn6fd6l2ay.py
# Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer2___0___pool => avg_pool2d_1
triton_poi_fused_avg_pool2d_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 326144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x3 = (xindex // 40768)
    x6 = (xindex // 28) % 1456
    x7 = xindex % 40768
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (489159 + (2*x0) + (112*x6) + (652288*x3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (489160 + (2*x0) + (112*x6) + (652288*x3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (489161 + (2*x0) + (112*x6) + (652288*x3)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (489215 + (2*x0) + (112*x6) + (652288*x3)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (489216 + (2*x0) + (112*x6) + (652288*x3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (489217 + (2*x0) + (112*x6) + (652288*x3)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (489271 + (2*x0) + (112*x6) + (652288*x3)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (489272 + (2*x0) + (112*x6) + (652288*x3)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (489273 + (2*x0) + (112*x6) + (652288*x3)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 57, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x7 + (163072*x3)), tmp145, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gn/cgnjmt5andx3wojt5yf52ptxnk5u4kgegijzdxz4sm7ptvsu7722.py
# Source Nodes: [sp_41, sp_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_41 => add_44, mul_55, mul_56, sub_18
# sp_42 => relu_17
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 326144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 52
    x2 = (xindex // 40768)
    x4 = xindex % 40768
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x4 + (163072*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/67/c67kfzbykiailczwhcmwwlmhuixp33b7yje6ddxjmm2olfquvxrh.py
# Source Nodes: [out_28], Original ATen: [aten.convolution]
# out_28 => convolution_21
triton_poi_fused_convolution_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (163072*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4o/c4oao4usvtye73fhyp7dfawcac54gpubikxtrugavk6sddnfsbc5.py
# Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_29 => add_50, mul_64, mul_65, sub_21
# out_30 => add_53
# shortcut_5 => add_52, mul_67, mul_68, sub_22
# shortcut_6 => relu_20
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 784
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
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
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (512*x2) + (401408*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sudscwgumogxl2oeps57gybogzc3gnnmtaz6nmpahonnljiwdj.py
# Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_33 => add_55, mul_70, mul_71, sub_23
# out_34 => relu_21
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1304576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 208
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sg/csgtae4hui6yxosdwa4wugdlmxdcg2gibkfd6ryx6jxojduzvjvr.py
# Source Nodes: [sp_53], Original ATen: [aten.convolution]
# sp_53 => convolution_24
triton_poi_fused_convolution_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y0) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (52*x2) + (40768*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4myle7keyuysmstkngj3gbsfvucscj4vex64tv6x2bvcwaron4.py
# Source Nodes: [sp_54, sp_55, sp_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_54 => add_57, mul_73, mul_74, sub_24
# sp_55 => relu_22
# sp_56 => add_58
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (40768 + x2 + (784*y0) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (784*y0) + (163072*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (52*x2) + (40768*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3a5zraohiegz4tgnblqdirmzudtcoz5htdpzd7pc7wat4ezhh2r.py
# Source Nodes: [sp_58, sp_59, sp_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_58 => add_60, mul_76, mul_77, sub_25
# sp_59 => relu_23
# sp_60 => add_61
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (81536 + x2 + (784*y0) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (784*y0) + (163072*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (52*x2) + (40768*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceu2qeu7zlestruwihxbrwmrui7fgbeguqllwtwu3fpdrr7xarta.py
# Source Nodes: [cat_61], Original ATen: [aten.cat]
# cat_61 => cat_4
triton_poi_fused_cat_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 326144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40768
    x1 = (xindex // 40768)
    tmp0 = tl.load(in_ptr0 + (122304 + x0 + (163072*x1)), xmask)
    tl.store(out_ptr0 + (x0 + (163072*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w4/cw4xyf37abejfxr3ouqzc4jvqrfaoed63b4z7wo2fifw4nx62hvs.py
# Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_37 => add_65, mul_82, mul_83, sub_27
# out_38 => add_66
# shortcut_7 => relu_25
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iy/ciyqdeitdonh4cork3gkiwtqsdbvsxiz3wljk6uqsn4mcr6jpwb5.py
# Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_57 => add_94, mul_115, mul_116, sub_38
# out_58 => relu_36
triton_poi_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2609152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 416
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xi/cxir5mwfpssevswlx5fkkdgvqfpvxqvk7jqslylgz4qneod5g6bl.py
# Source Nodes: [sp_92], Original ATen: [aten.convolution]
# sp_92 => convolution_39
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y0) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (81536*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7fciozqbipefheo7i56oksdngs2htuwbjwlwbtrdp2rucn7i6v.py
# Source Nodes: [sp_92], Original ATen: [aten.convolution]
# sp_92 => convolution_39
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10816
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (936*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csqblnuh4dcnb2jcw7etzomk64njp7q4sgviu7hh5gzm3ui2geey.py
# Source Nodes: [sp_96], Original ATen: [aten.convolution]
# sp_96 => convolution_40
triton_poi_fused_convolution_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (81536 + x2 + (784*y0) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (81536*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c656ehwfvodqiarwbshwtfwdr2wpow37c7kftpoif6ja4qvfgfjw.py
# Source Nodes: [sp_100], Original ATen: [aten.convolution]
# sp_100 => convolution_41
triton_poi_fused_convolution_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (163072 + x2 + (784*y0) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (81536*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4s/c4s3bcckl256ltrabhe6nuzxdbsou5clqe5pn4aun67r3lh5il7e.py
# Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer3___0___pool => avg_pool2d_2
triton_poi_fused_avg_pool2d_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x3 = (xindex // 20384)
    x6 = (xindex // 14) % 1456
    x7 = xindex % 20384
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (244579 + (2*x0) + (56*x6) + (326144*x3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (244580 + (2*x0) + (56*x6) + (326144*x3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (244581 + (2*x0) + (56*x6) + (326144*x3)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (244607 + (2*x0) + (56*x6) + (326144*x3)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (244608 + (2*x0) + (56*x6) + (326144*x3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (244609 + (2*x0) + (56*x6) + (326144*x3)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (244635 + (2*x0) + (56*x6) + (326144*x3)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (244636 + (2*x0) + (56*x6) + (326144*x3)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (244637 + (2*x0) + (56*x6) + (326144*x3)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 29, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x7 + (81536*x3)), tmp145, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eo/ceovh5b7lrpltpm4ut2uivlbvgg2q5kyt44ie5j36ngbflnywj5e.py
# Source Nodes: [sp_93, sp_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_93 => add_96, mul_118, mul_119, sub_39
# sp_94 => relu_37
triton_poi_fused__native_batch_norm_legit_no_training_relu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 104
    x2 = (xindex // 20384)
    x4 = xindex % 20384
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x4 + (81536*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ny/cnyk5sedgmsuzbozjwjxturu4zowqe5dhvq7lhgev5y4j32gl3pp.py
# Source Nodes: [out_60], Original ATen: [aten.convolution]
# out_60 => convolution_42
triton_poi_fused_convolution_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3328
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 416
    y1 = (yindex // 416)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (416*x2) + (81536*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pi/cpijycdbz7futq5wcqwmbxh6r7u5gywlk4gssz3ffsst3airn56b.py
# Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_61 => add_102, mul_127, mul_128, sub_42
# out_62 => add_105
# shortcut_10 => add_104, mul_130, mul_131, sub_43
# shortcut_11 => relu_40
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 196
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
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
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (1024*x2) + (200704*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2s2gbjeby73ysafj7x3yriargjheavmohtaqnswp6a63h5jjqi.py
# Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_65 => add_107, mul_133, mul_134, sub_44
# out_66 => relu_41
triton_poi_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 416
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwsbz74iji52b4xk74e2i5ti3qegz7lxliwq6glcnxslmczfvvr.py
# Source Nodes: [sp_105], Original ATen: [aten.convolution]
# sp_105 => convolution_45
triton_poi_fused_convolution_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (20384*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cuggecb7httveytv2gi4w5oflroamotgdmkfls4gejtgbdpu2jp6.py
# Source Nodes: [sp_106, sp_107, sp_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_106 => add_109, mul_136, mul_137, sub_45
# sp_107 => relu_42
# sp_108 => add_110
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (20384 + x2 + (196*y0) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (196*y0) + (81536*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (104*x2) + (20384*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4o/c4ootvpby5k3kj7fpnblsw62kx5lrmbxfe56hcv2ytcf3gsn5fz5.py
# Source Nodes: [sp_110, sp_111, sp_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_110 => add_112, mul_139, mul_140, sub_46
# sp_111 => relu_43
# sp_112 => add_113
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (40768 + x2 + (196*y0) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (196*y0) + (81536*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (104*x2) + (20384*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfxmgvcz5ggtnocoalmhotfamiddqqhocck6i5ysfec2y3u7lx3.py
# Source Nodes: [cat_57], Original ATen: [aten.cat]
# cat_57 => cat_8
triton_poi_fused_cat_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 20384
    x1 = (xindex // 20384)
    tmp0 = tl.load(in_ptr0 + (61152 + x0 + (81536*x1)), xmask)
    tl.store(out_ptr0 + (x0 + (81536*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a4/ca42kkadw6hu6eysbpedvpnehq2ixzctpieelarh2ddlzvtnrswf.py
# Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_69 => add_117, mul_145, mul_146, sub_48
# out_70 => add_118
# shortcut_12 => relu_45
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (1024*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/csw7hqwoquxxvljrktxmfjvtq7xx3zfcb2ie227zkv5klpcla4ea.py
# Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_241 => add_393, mul_463, mul_464, sub_154
# out_242 => relu_151
triton_poi_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1304576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 832
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43mksm4zunf6i6dnzof2evtf5h75nlnttcw2ikywvckaicwhyhx.py
# Source Nodes: [sp_391], Original ATen: [aten.convolution]
# sp_391 => convolution_155
triton_poi_fused_convolution_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (40768*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/caw7asrqa6md22cm272skpoowrxxur4zwqdpceabvs6hgxrq4z2q.py
# Source Nodes: [sp_391], Original ATen: [aten.convolution]
# sp_391 => convolution_155
triton_poi_fused_convolution_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 43264
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (1872*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gi/cgixmr42l4lkdtx52uysdslsaj6kyozg6vt3wj42cdglxpl6x6do.py
# Source Nodes: [sp_395], Original ATen: [aten.convolution]
# sp_395 => convolution_156
triton_poi_fused_convolution_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (40768 + x2 + (196*y0) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (40768*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cag3aog5tbzspvgy5us5vlgtyl7vydhirxisgywjse65aipp3k6n.py
# Source Nodes: [sp_399], Original ATen: [aten.convolution]
# sp_399 => convolution_157
triton_poi_fused_convolution_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (81536 + x2 + (196*y0) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (40768*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jnuxmjtll3ywqxxdfvswk3wwvnm7rlmbsbms2myltoeetpw7b3.py
# Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer4___0___pool => avg_pool2d_3
triton_poi_fused_avg_pool2d_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 7) % 7
    x0 = xindex % 7
    x3 = (xindex // 10192)
    x6 = (xindex // 7) % 1456
    x7 = xindex % 10192
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (122289 + (2*x0) + (28*x6) + (163072*x3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (122290 + (2*x0) + (28*x6) + (163072*x3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (122291 + (2*x0) + (28*x6) + (163072*x3)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (122303 + (2*x0) + (28*x6) + (163072*x3)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (122304 + (2*x0) + (28*x6) + (163072*x3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (122305 + (2*x0) + (28*x6) + (163072*x3)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (122317 + (2*x0) + (28*x6) + (163072*x3)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (122318 + (2*x0) + (28*x6) + (163072*x3)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (122319 + (2*x0) + (28*x6) + (163072*x3)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 15, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x7 + (40768*x3)), tmp145, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/su/csuqfdwn4lvkp3whg6gwxihh7arufluq3xlqfhtbew2y367fh2sn.py
# Source Nodes: [sp_392, sp_393], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_392 => add_395, mul_466, mul_467, sub_155
# sp_393 => relu_152
triton_poi_fused__native_batch_norm_legit_no_training_relu_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 208
    x2 = (xindex // 10192)
    x4 = xindex % 10192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x4 + (40768*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnair3walpe2wiyabwxe7delvjpofd4ns6e6hure6aqsj534b2q.py
# Source Nodes: [out_244], Original ATen: [aten.convolution]
# out_244 => convolution_158
triton_poi_fused_convolution_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6656
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 832
    y1 = (yindex // 832)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (832*x2) + (40768*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5csdse4e3wuiawrc6elnkeebul57xuniypk6hh2h3pomhmey2j.py
# Source Nodes: [out_245, out_246, shortcut_34, shortcut_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_245 => add_401, mul_475, mul_476, sub_158
# out_246 => add_404
# shortcut_34 => add_403, mul_478, mul_479, sub_159
# shortcut_35 => relu_155
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_55', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 49
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
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
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (2048*x2) + (100352*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5cixjlanuhm44uw7mbjbwuhrifaq6nuicz4pnybzctqj2oyq6o.py
# Source Nodes: [out_249, out_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_249 => add_406, mul_481, mul_482, sub_160
# out_250 => relu_156
triton_poi_fused__native_batch_norm_legit_no_training_relu_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 326144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 832
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b7/cb7lyr2ulilfev3wnnjrkwlagxg3fhclqufctg5bryktjfywxq4o.py
# Source Nodes: [sp_404], Original ATen: [aten.convolution]
# sp_404 => convolution_161
triton_poi_fused_convolution_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y0) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (10192*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4uatj25oc237pbiojwmgzjuxf35s4fxf4x3xsc5r7dza3h3h3i2.py
# Source Nodes: [sp_405, sp_406, sp_407], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_405 => add_408, mul_484, mul_485, sub_161
# sp_406 => relu_157
# sp_407 => add_409
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (10192 + x2 + (49*y0) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (49*y0) + (40768*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (208*x2) + (10192*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clzze67tq4oempibpge5nltwbbid34rebmy7sygpy54doan6rhye.py
# Source Nodes: [sp_409, sp_410, sp_411], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# sp_409 => add_411, mul_487, mul_488, sub_162
# sp_410 => relu_158
# sp_411 => add_412
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (20384 + x2 + (49*y0) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (49*y0) + (40768*y1)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (208*x2) + (10192*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vvwzqqbypbgwewhnxkdacuhmrezxexvhz62vwcq4autidmgws2.py
# Source Nodes: [cat_34], Original ATen: [aten.cat]
# cat_34 => cat_31
triton_poi_fused_cat_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 10192
    x1 = (xindex // 10192)
    tmp0 = tl.load(in_ptr0 + (30576 + x0 + (40768*x1)), xmask)
    tl.store(out_ptr0 + (x0 + (40768*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nu/cnub4pgijubpjeyf7lxtinlctamtzut7jkvhwufprr6v24qravcs.py
# Source Nodes: [out_253, out_254, shortcut_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_253 => add_416, mul_493, mul_494, sub_164
# out_254 => add_417
# shortcut_36 => relu_160
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (100352*y1)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (2048*y3)), tmp17, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpudf4sxrctgjdhyaqejwv5yvk44kzdddypue46kvn6et6najlkl.py
# Source Nodes: [out_261, out_262, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
# out_261 => add_429, mul_508, mul_509, sub_169
# out_262 => add_430
# x_8 => relu_165
# x_9 => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_62', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0 + (2048*r2) + (100352*x1)), rmask, other=0.0)
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
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (104, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg4_1, (104, ), (1, ))
    assert_size_stride(arg5_1, (104, ), (1, ))
    assert_size_stride(arg6_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg7_1, (26, ), (1, ))
    assert_size_stride(arg8_1, (26, ), (1, ))
    assert_size_stride(arg9_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg10_1, (26, ), (1, ))
    assert_size_stride(arg11_1, (26, ), (1, ))
    assert_size_stride(arg12_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg13_1, (26, ), (1, ))
    assert_size_stride(arg14_1, (26, ), (1, ))
    assert_size_stride(arg15_1, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (104, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg22_1, (104, ), (1, ))
    assert_size_stride(arg23_1, (104, ), (1, ))
    assert_size_stride(arg24_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg25_1, (26, ), (1, ))
    assert_size_stride(arg26_1, (26, ), (1, ))
    assert_size_stride(arg27_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg28_1, (26, ), (1, ))
    assert_size_stride(arg29_1, (26, ), (1, ))
    assert_size_stride(arg30_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg31_1, (26, ), (1, ))
    assert_size_stride(arg32_1, (26, ), (1, ))
    assert_size_stride(arg33_1, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (104, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg37_1, (104, ), (1, ))
    assert_size_stride(arg38_1, (104, ), (1, ))
    assert_size_stride(arg39_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg40_1, (26, ), (1, ))
    assert_size_stride(arg41_1, (26, ), (1, ))
    assert_size_stride(arg42_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg43_1, (26, ), (1, ))
    assert_size_stride(arg44_1, (26, ), (1, ))
    assert_size_stride(arg45_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg46_1, (26, ), (1, ))
    assert_size_stride(arg47_1, (26, ), (1, ))
    assert_size_stride(arg48_1, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (208, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg52_1, (208, ), (1, ))
    assert_size_stride(arg53_1, (208, ), (1, ))
    assert_size_stride(arg54_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg55_1, (52, ), (1, ))
    assert_size_stride(arg56_1, (52, ), (1, ))
    assert_size_stride(arg57_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg58_1, (52, ), (1, ))
    assert_size_stride(arg59_1, (52, ), (1, ))
    assert_size_stride(arg60_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg61_1, (52, ), (1, ))
    assert_size_stride(arg62_1, (52, ), (1, ))
    assert_size_stride(arg63_1, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg70_1, (208, ), (1, ))
    assert_size_stride(arg71_1, (208, ), (1, ))
    assert_size_stride(arg72_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg73_1, (52, ), (1, ))
    assert_size_stride(arg74_1, (52, ), (1, ))
    assert_size_stride(arg75_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg76_1, (52, ), (1, ))
    assert_size_stride(arg77_1, (52, ), (1, ))
    assert_size_stride(arg78_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg79_1, (52, ), (1, ))
    assert_size_stride(arg80_1, (52, ), (1, ))
    assert_size_stride(arg81_1, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg85_1, (208, ), (1, ))
    assert_size_stride(arg86_1, (208, ), (1, ))
    assert_size_stride(arg87_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg88_1, (52, ), (1, ))
    assert_size_stride(arg89_1, (52, ), (1, ))
    assert_size_stride(arg90_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg91_1, (52, ), (1, ))
    assert_size_stride(arg92_1, (52, ), (1, ))
    assert_size_stride(arg93_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg94_1, (52, ), (1, ))
    assert_size_stride(arg95_1, (52, ), (1, ))
    assert_size_stride(arg96_1, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg100_1, (208, ), (1, ))
    assert_size_stride(arg101_1, (208, ), (1, ))
    assert_size_stride(arg102_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg103_1, (52, ), (1, ))
    assert_size_stride(arg104_1, (52, ), (1, ))
    assert_size_stride(arg105_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg106_1, (52, ), (1, ))
    assert_size_stride(arg107_1, (52, ), (1, ))
    assert_size_stride(arg108_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg109_1, (52, ), (1, ))
    assert_size_stride(arg110_1, (52, ), (1, ))
    assert_size_stride(arg111_1, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (416, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg115_1, (416, ), (1, ))
    assert_size_stride(arg116_1, (416, ), (1, ))
    assert_size_stride(arg117_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg118_1, (104, ), (1, ))
    assert_size_stride(arg119_1, (104, ), (1, ))
    assert_size_stride(arg120_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg121_1, (104, ), (1, ))
    assert_size_stride(arg122_1, (104, ), (1, ))
    assert_size_stride(arg123_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg124_1, (104, ), (1, ))
    assert_size_stride(arg125_1, (104, ), (1, ))
    assert_size_stride(arg126_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg133_1, (416, ), (1, ))
    assert_size_stride(arg134_1, (416, ), (1, ))
    assert_size_stride(arg135_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg136_1, (104, ), (1, ))
    assert_size_stride(arg137_1, (104, ), (1, ))
    assert_size_stride(arg138_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg139_1, (104, ), (1, ))
    assert_size_stride(arg140_1, (104, ), (1, ))
    assert_size_stride(arg141_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg142_1, (104, ), (1, ))
    assert_size_stride(arg143_1, (104, ), (1, ))
    assert_size_stride(arg144_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg148_1, (416, ), (1, ))
    assert_size_stride(arg149_1, (416, ), (1, ))
    assert_size_stride(arg150_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg151_1, (104, ), (1, ))
    assert_size_stride(arg152_1, (104, ), (1, ))
    assert_size_stride(arg153_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg154_1, (104, ), (1, ))
    assert_size_stride(arg155_1, (104, ), (1, ))
    assert_size_stride(arg156_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg157_1, (104, ), (1, ))
    assert_size_stride(arg158_1, (104, ), (1, ))
    assert_size_stride(arg159_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg163_1, (416, ), (1, ))
    assert_size_stride(arg164_1, (416, ), (1, ))
    assert_size_stride(arg165_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg166_1, (104, ), (1, ))
    assert_size_stride(arg167_1, (104, ), (1, ))
    assert_size_stride(arg168_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg169_1, (104, ), (1, ))
    assert_size_stride(arg170_1, (104, ), (1, ))
    assert_size_stride(arg171_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg172_1, (104, ), (1, ))
    assert_size_stride(arg173_1, (104, ), (1, ))
    assert_size_stride(arg174_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg178_1, (416, ), (1, ))
    assert_size_stride(arg179_1, (416, ), (1, ))
    assert_size_stride(arg180_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg181_1, (104, ), (1, ))
    assert_size_stride(arg182_1, (104, ), (1, ))
    assert_size_stride(arg183_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg184_1, (104, ), (1, ))
    assert_size_stride(arg185_1, (104, ), (1, ))
    assert_size_stride(arg186_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg187_1, (104, ), (1, ))
    assert_size_stride(arg188_1, (104, ), (1, ))
    assert_size_stride(arg189_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg193_1, (416, ), (1, ))
    assert_size_stride(arg194_1, (416, ), (1, ))
    assert_size_stride(arg195_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg196_1, (104, ), (1, ))
    assert_size_stride(arg197_1, (104, ), (1, ))
    assert_size_stride(arg198_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg199_1, (104, ), (1, ))
    assert_size_stride(arg200_1, (104, ), (1, ))
    assert_size_stride(arg201_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg202_1, (104, ), (1, ))
    assert_size_stride(arg203_1, (104, ), (1, ))
    assert_size_stride(arg204_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg208_1, (416, ), (1, ))
    assert_size_stride(arg209_1, (416, ), (1, ))
    assert_size_stride(arg210_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg211_1, (104, ), (1, ))
    assert_size_stride(arg212_1, (104, ), (1, ))
    assert_size_stride(arg213_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg214_1, (104, ), (1, ))
    assert_size_stride(arg215_1, (104, ), (1, ))
    assert_size_stride(arg216_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg217_1, (104, ), (1, ))
    assert_size_stride(arg218_1, (104, ), (1, ))
    assert_size_stride(arg219_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg220_1, (1024, ), (1, ))
    assert_size_stride(arg221_1, (1024, ), (1, ))
    assert_size_stride(arg222_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg223_1, (416, ), (1, ))
    assert_size_stride(arg224_1, (416, ), (1, ))
    assert_size_stride(arg225_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg226_1, (104, ), (1, ))
    assert_size_stride(arg227_1, (104, ), (1, ))
    assert_size_stride(arg228_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg229_1, (104, ), (1, ))
    assert_size_stride(arg230_1, (104, ), (1, ))
    assert_size_stride(arg231_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg232_1, (104, ), (1, ))
    assert_size_stride(arg233_1, (104, ), (1, ))
    assert_size_stride(arg234_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, ), (1, ))
    assert_size_stride(arg237_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg238_1, (416, ), (1, ))
    assert_size_stride(arg239_1, (416, ), (1, ))
    assert_size_stride(arg240_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg241_1, (104, ), (1, ))
    assert_size_stride(arg242_1, (104, ), (1, ))
    assert_size_stride(arg243_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg244_1, (104, ), (1, ))
    assert_size_stride(arg245_1, (104, ), (1, ))
    assert_size_stride(arg246_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg247_1, (104, ), (1, ))
    assert_size_stride(arg248_1, (104, ), (1, ))
    assert_size_stride(arg249_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (1024, ), (1, ))
    assert_size_stride(arg252_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg253_1, (416, ), (1, ))
    assert_size_stride(arg254_1, (416, ), (1, ))
    assert_size_stride(arg255_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg256_1, (104, ), (1, ))
    assert_size_stride(arg257_1, (104, ), (1, ))
    assert_size_stride(arg258_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg259_1, (104, ), (1, ))
    assert_size_stride(arg260_1, (104, ), (1, ))
    assert_size_stride(arg261_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg262_1, (104, ), (1, ))
    assert_size_stride(arg263_1, (104, ), (1, ))
    assert_size_stride(arg264_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg265_1, (1024, ), (1, ))
    assert_size_stride(arg266_1, (1024, ), (1, ))
    assert_size_stride(arg267_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg268_1, (416, ), (1, ))
    assert_size_stride(arg269_1, (416, ), (1, ))
    assert_size_stride(arg270_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg271_1, (104, ), (1, ))
    assert_size_stride(arg272_1, (104, ), (1, ))
    assert_size_stride(arg273_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg274_1, (104, ), (1, ))
    assert_size_stride(arg275_1, (104, ), (1, ))
    assert_size_stride(arg276_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg277_1, (104, ), (1, ))
    assert_size_stride(arg278_1, (104, ), (1, ))
    assert_size_stride(arg279_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg283_1, (416, ), (1, ))
    assert_size_stride(arg284_1, (416, ), (1, ))
    assert_size_stride(arg285_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg286_1, (104, ), (1, ))
    assert_size_stride(arg287_1, (104, ), (1, ))
    assert_size_stride(arg288_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg289_1, (104, ), (1, ))
    assert_size_stride(arg290_1, (104, ), (1, ))
    assert_size_stride(arg291_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg292_1, (104, ), (1, ))
    assert_size_stride(arg293_1, (104, ), (1, ))
    assert_size_stride(arg294_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg295_1, (1024, ), (1, ))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg298_1, (416, ), (1, ))
    assert_size_stride(arg299_1, (416, ), (1, ))
    assert_size_stride(arg300_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg301_1, (104, ), (1, ))
    assert_size_stride(arg302_1, (104, ), (1, ))
    assert_size_stride(arg303_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg304_1, (104, ), (1, ))
    assert_size_stride(arg305_1, (104, ), (1, ))
    assert_size_stride(arg306_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg307_1, (104, ), (1, ))
    assert_size_stride(arg308_1, (104, ), (1, ))
    assert_size_stride(arg309_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, ), (1, ))
    assert_size_stride(arg312_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg313_1, (416, ), (1, ))
    assert_size_stride(arg314_1, (416, ), (1, ))
    assert_size_stride(arg315_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg316_1, (104, ), (1, ))
    assert_size_stride(arg317_1, (104, ), (1, ))
    assert_size_stride(arg318_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg319_1, (104, ), (1, ))
    assert_size_stride(arg320_1, (104, ), (1, ))
    assert_size_stride(arg321_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg322_1, (104, ), (1, ))
    assert_size_stride(arg323_1, (104, ), (1, ))
    assert_size_stride(arg324_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg325_1, (1024, ), (1, ))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg328_1, (416, ), (1, ))
    assert_size_stride(arg329_1, (416, ), (1, ))
    assert_size_stride(arg330_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg331_1, (104, ), (1, ))
    assert_size_stride(arg332_1, (104, ), (1, ))
    assert_size_stride(arg333_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg334_1, (104, ), (1, ))
    assert_size_stride(arg335_1, (104, ), (1, ))
    assert_size_stride(arg336_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg337_1, (104, ), (1, ))
    assert_size_stride(arg338_1, (104, ), (1, ))
    assert_size_stride(arg339_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg340_1, (1024, ), (1, ))
    assert_size_stride(arg341_1, (1024, ), (1, ))
    assert_size_stride(arg342_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg343_1, (416, ), (1, ))
    assert_size_stride(arg344_1, (416, ), (1, ))
    assert_size_stride(arg345_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg346_1, (104, ), (1, ))
    assert_size_stride(arg347_1, (104, ), (1, ))
    assert_size_stride(arg348_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg349_1, (104, ), (1, ))
    assert_size_stride(arg350_1, (104, ), (1, ))
    assert_size_stride(arg351_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg352_1, (104, ), (1, ))
    assert_size_stride(arg353_1, (104, ), (1, ))
    assert_size_stride(arg354_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg355_1, (1024, ), (1, ))
    assert_size_stride(arg356_1, (1024, ), (1, ))
    assert_size_stride(arg357_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg358_1, (416, ), (1, ))
    assert_size_stride(arg359_1, (416, ), (1, ))
    assert_size_stride(arg360_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg361_1, (104, ), (1, ))
    assert_size_stride(arg362_1, (104, ), (1, ))
    assert_size_stride(arg363_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg364_1, (104, ), (1, ))
    assert_size_stride(arg365_1, (104, ), (1, ))
    assert_size_stride(arg366_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg367_1, (104, ), (1, ))
    assert_size_stride(arg368_1, (104, ), (1, ))
    assert_size_stride(arg369_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg370_1, (1024, ), (1, ))
    assert_size_stride(arg371_1, (1024, ), (1, ))
    assert_size_stride(arg372_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg373_1, (416, ), (1, ))
    assert_size_stride(arg374_1, (416, ), (1, ))
    assert_size_stride(arg375_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg376_1, (104, ), (1, ))
    assert_size_stride(arg377_1, (104, ), (1, ))
    assert_size_stride(arg378_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg379_1, (104, ), (1, ))
    assert_size_stride(arg380_1, (104, ), (1, ))
    assert_size_stride(arg381_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg382_1, (104, ), (1, ))
    assert_size_stride(arg383_1, (104, ), (1, ))
    assert_size_stride(arg384_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg385_1, (1024, ), (1, ))
    assert_size_stride(arg386_1, (1024, ), (1, ))
    assert_size_stride(arg387_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg388_1, (416, ), (1, ))
    assert_size_stride(arg389_1, (416, ), (1, ))
    assert_size_stride(arg390_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg391_1, (104, ), (1, ))
    assert_size_stride(arg392_1, (104, ), (1, ))
    assert_size_stride(arg393_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg394_1, (104, ), (1, ))
    assert_size_stride(arg395_1, (104, ), (1, ))
    assert_size_stride(arg396_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg397_1, (104, ), (1, ))
    assert_size_stride(arg398_1, (104, ), (1, ))
    assert_size_stride(arg399_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg400_1, (1024, ), (1, ))
    assert_size_stride(arg401_1, (1024, ), (1, ))
    assert_size_stride(arg402_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg403_1, (416, ), (1, ))
    assert_size_stride(arg404_1, (416, ), (1, ))
    assert_size_stride(arg405_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg406_1, (104, ), (1, ))
    assert_size_stride(arg407_1, (104, ), (1, ))
    assert_size_stride(arg408_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg409_1, (104, ), (1, ))
    assert_size_stride(arg410_1, (104, ), (1, ))
    assert_size_stride(arg411_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg412_1, (104, ), (1, ))
    assert_size_stride(arg413_1, (104, ), (1, ))
    assert_size_stride(arg414_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg415_1, (1024, ), (1, ))
    assert_size_stride(arg416_1, (1024, ), (1, ))
    assert_size_stride(arg417_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg418_1, (416, ), (1, ))
    assert_size_stride(arg419_1, (416, ), (1, ))
    assert_size_stride(arg420_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg421_1, (104, ), (1, ))
    assert_size_stride(arg422_1, (104, ), (1, ))
    assert_size_stride(arg423_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg424_1, (104, ), (1, ))
    assert_size_stride(arg425_1, (104, ), (1, ))
    assert_size_stride(arg426_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg427_1, (104, ), (1, ))
    assert_size_stride(arg428_1, (104, ), (1, ))
    assert_size_stride(arg429_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg430_1, (1024, ), (1, ))
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg433_1, (416, ), (1, ))
    assert_size_stride(arg434_1, (416, ), (1, ))
    assert_size_stride(arg435_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg436_1, (104, ), (1, ))
    assert_size_stride(arg437_1, (104, ), (1, ))
    assert_size_stride(arg438_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg439_1, (104, ), (1, ))
    assert_size_stride(arg440_1, (104, ), (1, ))
    assert_size_stride(arg441_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg442_1, (104, ), (1, ))
    assert_size_stride(arg443_1, (104, ), (1, ))
    assert_size_stride(arg444_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg445_1, (1024, ), (1, ))
    assert_size_stride(arg446_1, (1024, ), (1, ))
    assert_size_stride(arg447_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg448_1, (416, ), (1, ))
    assert_size_stride(arg449_1, (416, ), (1, ))
    assert_size_stride(arg450_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg451_1, (104, ), (1, ))
    assert_size_stride(arg452_1, (104, ), (1, ))
    assert_size_stride(arg453_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg454_1, (104, ), (1, ))
    assert_size_stride(arg455_1, (104, ), (1, ))
    assert_size_stride(arg456_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg457_1, (104, ), (1, ))
    assert_size_stride(arg458_1, (104, ), (1, ))
    assert_size_stride(arg459_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg460_1, (1024, ), (1, ))
    assert_size_stride(arg461_1, (1024, ), (1, ))
    assert_size_stride(arg462_1, (832, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg463_1, (832, ), (1, ))
    assert_size_stride(arg464_1, (832, ), (1, ))
    assert_size_stride(arg465_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg466_1, (208, ), (1, ))
    assert_size_stride(arg467_1, (208, ), (1, ))
    assert_size_stride(arg468_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg469_1, (208, ), (1, ))
    assert_size_stride(arg470_1, (208, ), (1, ))
    assert_size_stride(arg471_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg472_1, (208, ), (1, ))
    assert_size_stride(arg473_1, (208, ), (1, ))
    assert_size_stride(arg474_1, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg475_1, (2048, ), (1, ))
    assert_size_stride(arg476_1, (2048, ), (1, ))
    assert_size_stride(arg477_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg478_1, (2048, ), (1, ))
    assert_size_stride(arg479_1, (2048, ), (1, ))
    assert_size_stride(arg480_1, (832, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg481_1, (832, ), (1, ))
    assert_size_stride(arg482_1, (832, ), (1, ))
    assert_size_stride(arg483_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg484_1, (208, ), (1, ))
    assert_size_stride(arg485_1, (208, ), (1, ))
    assert_size_stride(arg486_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg487_1, (208, ), (1, ))
    assert_size_stride(arg488_1, (208, ), (1, ))
    assert_size_stride(arg489_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg490_1, (208, ), (1, ))
    assert_size_stride(arg491_1, (208, ), (1, ))
    assert_size_stride(arg492_1, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg493_1, (2048, ), (1, ))
    assert_size_stride(arg494_1, (2048, ), (1, ))
    assert_size_stride(arg495_1, (832, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg496_1, (832, ), (1, ))
    assert_size_stride(arg497_1, (832, ), (1, ))
    assert_size_stride(arg498_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg499_1, (208, ), (1, ))
    assert_size_stride(arg500_1, (208, ), (1, ))
    assert_size_stride(arg501_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg502_1, (208, ), (1, ))
    assert_size_stride(arg503_1, (208, ), (1, ))
    assert_size_stride(arg504_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg505_1, (208, ), (1, ))
    assert_size_stride(arg506_1, (208, ), (1, ))
    assert_size_stride(arg507_1, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg508_1, (2048, ), (1, ))
    assert_size_stride(arg509_1, (2048, ), (1, ))
    assert_size_stride(arg510_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg511_1, (1000, ), (1, ))
    assert_size_stride(arg512_1, (64, ), (1, ))
    assert_size_stride(arg513_1, (64, ), (1, ))
    assert_size_stride(arg514_1, (), ())
    assert_size_stride(arg515_1, (104, ), (1, ))
    assert_size_stride(arg516_1, (104, ), (1, ))
    assert_size_stride(arg517_1, (), ())
    assert_size_stride(arg518_1, (26, ), (1, ))
    assert_size_stride(arg519_1, (26, ), (1, ))
    assert_size_stride(arg520_1, (), ())
    assert_size_stride(arg521_1, (26, ), (1, ))
    assert_size_stride(arg522_1, (26, ), (1, ))
    assert_size_stride(arg523_1, (), ())
    assert_size_stride(arg524_1, (26, ), (1, ))
    assert_size_stride(arg525_1, (26, ), (1, ))
    assert_size_stride(arg526_1, (), ())
    assert_size_stride(arg527_1, (256, ), (1, ))
    assert_size_stride(arg528_1, (256, ), (1, ))
    assert_size_stride(arg529_1, (), ())
    assert_size_stride(arg530_1, (256, ), (1, ))
    assert_size_stride(arg531_1, (256, ), (1, ))
    assert_size_stride(arg532_1, (), ())
    assert_size_stride(arg533_1, (104, ), (1, ))
    assert_size_stride(arg534_1, (104, ), (1, ))
    assert_size_stride(arg535_1, (), ())
    assert_size_stride(arg536_1, (26, ), (1, ))
    assert_size_stride(arg537_1, (26, ), (1, ))
    assert_size_stride(arg538_1, (), ())
    assert_size_stride(arg539_1, (26, ), (1, ))
    assert_size_stride(arg540_1, (26, ), (1, ))
    assert_size_stride(arg541_1, (), ())
    assert_size_stride(arg542_1, (26, ), (1, ))
    assert_size_stride(arg543_1, (26, ), (1, ))
    assert_size_stride(arg544_1, (), ())
    assert_size_stride(arg545_1, (256, ), (1, ))
    assert_size_stride(arg546_1, (256, ), (1, ))
    assert_size_stride(arg547_1, (), ())
    assert_size_stride(arg548_1, (104, ), (1, ))
    assert_size_stride(arg549_1, (104, ), (1, ))
    assert_size_stride(arg550_1, (), ())
    assert_size_stride(arg551_1, (26, ), (1, ))
    assert_size_stride(arg552_1, (26, ), (1, ))
    assert_size_stride(arg553_1, (), ())
    assert_size_stride(arg554_1, (26, ), (1, ))
    assert_size_stride(arg555_1, (26, ), (1, ))
    assert_size_stride(arg556_1, (), ())
    assert_size_stride(arg557_1, (26, ), (1, ))
    assert_size_stride(arg558_1, (26, ), (1, ))
    assert_size_stride(arg559_1, (), ())
    assert_size_stride(arg560_1, (256, ), (1, ))
    assert_size_stride(arg561_1, (256, ), (1, ))
    assert_size_stride(arg562_1, (), ())
    assert_size_stride(arg563_1, (208, ), (1, ))
    assert_size_stride(arg564_1, (208, ), (1, ))
    assert_size_stride(arg565_1, (), ())
    assert_size_stride(arg566_1, (52, ), (1, ))
    assert_size_stride(arg567_1, (52, ), (1, ))
    assert_size_stride(arg568_1, (), ())
    assert_size_stride(arg569_1, (52, ), (1, ))
    assert_size_stride(arg570_1, (52, ), (1, ))
    assert_size_stride(arg571_1, (), ())
    assert_size_stride(arg572_1, (52, ), (1, ))
    assert_size_stride(arg573_1, (52, ), (1, ))
    assert_size_stride(arg574_1, (), ())
    assert_size_stride(arg575_1, (512, ), (1, ))
    assert_size_stride(arg576_1, (512, ), (1, ))
    assert_size_stride(arg577_1, (), ())
    assert_size_stride(arg578_1, (512, ), (1, ))
    assert_size_stride(arg579_1, (512, ), (1, ))
    assert_size_stride(arg580_1, (), ())
    assert_size_stride(arg581_1, (208, ), (1, ))
    assert_size_stride(arg582_1, (208, ), (1, ))
    assert_size_stride(arg583_1, (), ())
    assert_size_stride(arg584_1, (52, ), (1, ))
    assert_size_stride(arg585_1, (52, ), (1, ))
    assert_size_stride(arg586_1, (), ())
    assert_size_stride(arg587_1, (52, ), (1, ))
    assert_size_stride(arg588_1, (52, ), (1, ))
    assert_size_stride(arg589_1, (), ())
    assert_size_stride(arg590_1, (52, ), (1, ))
    assert_size_stride(arg591_1, (52, ), (1, ))
    assert_size_stride(arg592_1, (), ())
    assert_size_stride(arg593_1, (512, ), (1, ))
    assert_size_stride(arg594_1, (512, ), (1, ))
    assert_size_stride(arg595_1, (), ())
    assert_size_stride(arg596_1, (208, ), (1, ))
    assert_size_stride(arg597_1, (208, ), (1, ))
    assert_size_stride(arg598_1, (), ())
    assert_size_stride(arg599_1, (52, ), (1, ))
    assert_size_stride(arg600_1, (52, ), (1, ))
    assert_size_stride(arg601_1, (), ())
    assert_size_stride(arg602_1, (52, ), (1, ))
    assert_size_stride(arg603_1, (52, ), (1, ))
    assert_size_stride(arg604_1, (), ())
    assert_size_stride(arg605_1, (52, ), (1, ))
    assert_size_stride(arg606_1, (52, ), (1, ))
    assert_size_stride(arg607_1, (), ())
    assert_size_stride(arg608_1, (512, ), (1, ))
    assert_size_stride(arg609_1, (512, ), (1, ))
    assert_size_stride(arg610_1, (), ())
    assert_size_stride(arg611_1, (208, ), (1, ))
    assert_size_stride(arg612_1, (208, ), (1, ))
    assert_size_stride(arg613_1, (), ())
    assert_size_stride(arg614_1, (52, ), (1, ))
    assert_size_stride(arg615_1, (52, ), (1, ))
    assert_size_stride(arg616_1, (), ())
    assert_size_stride(arg617_1, (52, ), (1, ))
    assert_size_stride(arg618_1, (52, ), (1, ))
    assert_size_stride(arg619_1, (), ())
    assert_size_stride(arg620_1, (52, ), (1, ))
    assert_size_stride(arg621_1, (52, ), (1, ))
    assert_size_stride(arg622_1, (), ())
    assert_size_stride(arg623_1, (512, ), (1, ))
    assert_size_stride(arg624_1, (512, ), (1, ))
    assert_size_stride(arg625_1, (), ())
    assert_size_stride(arg626_1, (416, ), (1, ))
    assert_size_stride(arg627_1, (416, ), (1, ))
    assert_size_stride(arg628_1, (), ())
    assert_size_stride(arg629_1, (104, ), (1, ))
    assert_size_stride(arg630_1, (104, ), (1, ))
    assert_size_stride(arg631_1, (), ())
    assert_size_stride(arg632_1, (104, ), (1, ))
    assert_size_stride(arg633_1, (104, ), (1, ))
    assert_size_stride(arg634_1, (), ())
    assert_size_stride(arg635_1, (104, ), (1, ))
    assert_size_stride(arg636_1, (104, ), (1, ))
    assert_size_stride(arg637_1, (), ())
    assert_size_stride(arg638_1, (1024, ), (1, ))
    assert_size_stride(arg639_1, (1024, ), (1, ))
    assert_size_stride(arg640_1, (), ())
    assert_size_stride(arg641_1, (1024, ), (1, ))
    assert_size_stride(arg642_1, (1024, ), (1, ))
    assert_size_stride(arg643_1, (), ())
    assert_size_stride(arg644_1, (416, ), (1, ))
    assert_size_stride(arg645_1, (416, ), (1, ))
    assert_size_stride(arg646_1, (), ())
    assert_size_stride(arg647_1, (104, ), (1, ))
    assert_size_stride(arg648_1, (104, ), (1, ))
    assert_size_stride(arg649_1, (), ())
    assert_size_stride(arg650_1, (104, ), (1, ))
    assert_size_stride(arg651_1, (104, ), (1, ))
    assert_size_stride(arg652_1, (), ())
    assert_size_stride(arg653_1, (104, ), (1, ))
    assert_size_stride(arg654_1, (104, ), (1, ))
    assert_size_stride(arg655_1, (), ())
    assert_size_stride(arg656_1, (1024, ), (1, ))
    assert_size_stride(arg657_1, (1024, ), (1, ))
    assert_size_stride(arg658_1, (), ())
    assert_size_stride(arg659_1, (416, ), (1, ))
    assert_size_stride(arg660_1, (416, ), (1, ))
    assert_size_stride(arg661_1, (), ())
    assert_size_stride(arg662_1, (104, ), (1, ))
    assert_size_stride(arg663_1, (104, ), (1, ))
    assert_size_stride(arg664_1, (), ())
    assert_size_stride(arg665_1, (104, ), (1, ))
    assert_size_stride(arg666_1, (104, ), (1, ))
    assert_size_stride(arg667_1, (), ())
    assert_size_stride(arg668_1, (104, ), (1, ))
    assert_size_stride(arg669_1, (104, ), (1, ))
    assert_size_stride(arg670_1, (), ())
    assert_size_stride(arg671_1, (1024, ), (1, ))
    assert_size_stride(arg672_1, (1024, ), (1, ))
    assert_size_stride(arg673_1, (), ())
    assert_size_stride(arg674_1, (416, ), (1, ))
    assert_size_stride(arg675_1, (416, ), (1, ))
    assert_size_stride(arg676_1, (), ())
    assert_size_stride(arg677_1, (104, ), (1, ))
    assert_size_stride(arg678_1, (104, ), (1, ))
    assert_size_stride(arg679_1, (), ())
    assert_size_stride(arg680_1, (104, ), (1, ))
    assert_size_stride(arg681_1, (104, ), (1, ))
    assert_size_stride(arg682_1, (), ())
    assert_size_stride(arg683_1, (104, ), (1, ))
    assert_size_stride(arg684_1, (104, ), (1, ))
    assert_size_stride(arg685_1, (), ())
    assert_size_stride(arg686_1, (1024, ), (1, ))
    assert_size_stride(arg687_1, (1024, ), (1, ))
    assert_size_stride(arg688_1, (), ())
    assert_size_stride(arg689_1, (416, ), (1, ))
    assert_size_stride(arg690_1, (416, ), (1, ))
    assert_size_stride(arg691_1, (), ())
    assert_size_stride(arg692_1, (104, ), (1, ))
    assert_size_stride(arg693_1, (104, ), (1, ))
    assert_size_stride(arg694_1, (), ())
    assert_size_stride(arg695_1, (104, ), (1, ))
    assert_size_stride(arg696_1, (104, ), (1, ))
    assert_size_stride(arg697_1, (), ())
    assert_size_stride(arg698_1, (104, ), (1, ))
    assert_size_stride(arg699_1, (104, ), (1, ))
    assert_size_stride(arg700_1, (), ())
    assert_size_stride(arg701_1, (1024, ), (1, ))
    assert_size_stride(arg702_1, (1024, ), (1, ))
    assert_size_stride(arg703_1, (), ())
    assert_size_stride(arg704_1, (416, ), (1, ))
    assert_size_stride(arg705_1, (416, ), (1, ))
    assert_size_stride(arg706_1, (), ())
    assert_size_stride(arg707_1, (104, ), (1, ))
    assert_size_stride(arg708_1, (104, ), (1, ))
    assert_size_stride(arg709_1, (), ())
    assert_size_stride(arg710_1, (104, ), (1, ))
    assert_size_stride(arg711_1, (104, ), (1, ))
    assert_size_stride(arg712_1, (), ())
    assert_size_stride(arg713_1, (104, ), (1, ))
    assert_size_stride(arg714_1, (104, ), (1, ))
    assert_size_stride(arg715_1, (), ())
    assert_size_stride(arg716_1, (1024, ), (1, ))
    assert_size_stride(arg717_1, (1024, ), (1, ))
    assert_size_stride(arg718_1, (), ())
    assert_size_stride(arg719_1, (416, ), (1, ))
    assert_size_stride(arg720_1, (416, ), (1, ))
    assert_size_stride(arg721_1, (), ())
    assert_size_stride(arg722_1, (104, ), (1, ))
    assert_size_stride(arg723_1, (104, ), (1, ))
    assert_size_stride(arg724_1, (), ())
    assert_size_stride(arg725_1, (104, ), (1, ))
    assert_size_stride(arg726_1, (104, ), (1, ))
    assert_size_stride(arg727_1, (), ())
    assert_size_stride(arg728_1, (104, ), (1, ))
    assert_size_stride(arg729_1, (104, ), (1, ))
    assert_size_stride(arg730_1, (), ())
    assert_size_stride(arg731_1, (1024, ), (1, ))
    assert_size_stride(arg732_1, (1024, ), (1, ))
    assert_size_stride(arg733_1, (), ())
    assert_size_stride(arg734_1, (416, ), (1, ))
    assert_size_stride(arg735_1, (416, ), (1, ))
    assert_size_stride(arg736_1, (), ())
    assert_size_stride(arg737_1, (104, ), (1, ))
    assert_size_stride(arg738_1, (104, ), (1, ))
    assert_size_stride(arg739_1, (), ())
    assert_size_stride(arg740_1, (104, ), (1, ))
    assert_size_stride(arg741_1, (104, ), (1, ))
    assert_size_stride(arg742_1, (), ())
    assert_size_stride(arg743_1, (104, ), (1, ))
    assert_size_stride(arg744_1, (104, ), (1, ))
    assert_size_stride(arg745_1, (), ())
    assert_size_stride(arg746_1, (1024, ), (1, ))
    assert_size_stride(arg747_1, (1024, ), (1, ))
    assert_size_stride(arg748_1, (), ())
    assert_size_stride(arg749_1, (416, ), (1, ))
    assert_size_stride(arg750_1, (416, ), (1, ))
    assert_size_stride(arg751_1, (), ())
    assert_size_stride(arg752_1, (104, ), (1, ))
    assert_size_stride(arg753_1, (104, ), (1, ))
    assert_size_stride(arg754_1, (), ())
    assert_size_stride(arg755_1, (104, ), (1, ))
    assert_size_stride(arg756_1, (104, ), (1, ))
    assert_size_stride(arg757_1, (), ())
    assert_size_stride(arg758_1, (104, ), (1, ))
    assert_size_stride(arg759_1, (104, ), (1, ))
    assert_size_stride(arg760_1, (), ())
    assert_size_stride(arg761_1, (1024, ), (1, ))
    assert_size_stride(arg762_1, (1024, ), (1, ))
    assert_size_stride(arg763_1, (), ())
    assert_size_stride(arg764_1, (416, ), (1, ))
    assert_size_stride(arg765_1, (416, ), (1, ))
    assert_size_stride(arg766_1, (), ())
    assert_size_stride(arg767_1, (104, ), (1, ))
    assert_size_stride(arg768_1, (104, ), (1, ))
    assert_size_stride(arg769_1, (), ())
    assert_size_stride(arg770_1, (104, ), (1, ))
    assert_size_stride(arg771_1, (104, ), (1, ))
    assert_size_stride(arg772_1, (), ())
    assert_size_stride(arg773_1, (104, ), (1, ))
    assert_size_stride(arg774_1, (104, ), (1, ))
    assert_size_stride(arg775_1, (), ())
    assert_size_stride(arg776_1, (1024, ), (1, ))
    assert_size_stride(arg777_1, (1024, ), (1, ))
    assert_size_stride(arg778_1, (), ())
    assert_size_stride(arg779_1, (416, ), (1, ))
    assert_size_stride(arg780_1, (416, ), (1, ))
    assert_size_stride(arg781_1, (), ())
    assert_size_stride(arg782_1, (104, ), (1, ))
    assert_size_stride(arg783_1, (104, ), (1, ))
    assert_size_stride(arg784_1, (), ())
    assert_size_stride(arg785_1, (104, ), (1, ))
    assert_size_stride(arg786_1, (104, ), (1, ))
    assert_size_stride(arg787_1, (), ())
    assert_size_stride(arg788_1, (104, ), (1, ))
    assert_size_stride(arg789_1, (104, ), (1, ))
    assert_size_stride(arg790_1, (), ())
    assert_size_stride(arg791_1, (1024, ), (1, ))
    assert_size_stride(arg792_1, (1024, ), (1, ))
    assert_size_stride(arg793_1, (), ())
    assert_size_stride(arg794_1, (416, ), (1, ))
    assert_size_stride(arg795_1, (416, ), (1, ))
    assert_size_stride(arg796_1, (), ())
    assert_size_stride(arg797_1, (104, ), (1, ))
    assert_size_stride(arg798_1, (104, ), (1, ))
    assert_size_stride(arg799_1, (), ())
    assert_size_stride(arg800_1, (104, ), (1, ))
    assert_size_stride(arg801_1, (104, ), (1, ))
    assert_size_stride(arg802_1, (), ())
    assert_size_stride(arg803_1, (104, ), (1, ))
    assert_size_stride(arg804_1, (104, ), (1, ))
    assert_size_stride(arg805_1, (), ())
    assert_size_stride(arg806_1, (1024, ), (1, ))
    assert_size_stride(arg807_1, (1024, ), (1, ))
    assert_size_stride(arg808_1, (), ())
    assert_size_stride(arg809_1, (416, ), (1, ))
    assert_size_stride(arg810_1, (416, ), (1, ))
    assert_size_stride(arg811_1, (), ())
    assert_size_stride(arg812_1, (104, ), (1, ))
    assert_size_stride(arg813_1, (104, ), (1, ))
    assert_size_stride(arg814_1, (), ())
    assert_size_stride(arg815_1, (104, ), (1, ))
    assert_size_stride(arg816_1, (104, ), (1, ))
    assert_size_stride(arg817_1, (), ())
    assert_size_stride(arg818_1, (104, ), (1, ))
    assert_size_stride(arg819_1, (104, ), (1, ))
    assert_size_stride(arg820_1, (), ())
    assert_size_stride(arg821_1, (1024, ), (1, ))
    assert_size_stride(arg822_1, (1024, ), (1, ))
    assert_size_stride(arg823_1, (), ())
    assert_size_stride(arg824_1, (416, ), (1, ))
    assert_size_stride(arg825_1, (416, ), (1, ))
    assert_size_stride(arg826_1, (), ())
    assert_size_stride(arg827_1, (104, ), (1, ))
    assert_size_stride(arg828_1, (104, ), (1, ))
    assert_size_stride(arg829_1, (), ())
    assert_size_stride(arg830_1, (104, ), (1, ))
    assert_size_stride(arg831_1, (104, ), (1, ))
    assert_size_stride(arg832_1, (), ())
    assert_size_stride(arg833_1, (104, ), (1, ))
    assert_size_stride(arg834_1, (104, ), (1, ))
    assert_size_stride(arg835_1, (), ())
    assert_size_stride(arg836_1, (1024, ), (1, ))
    assert_size_stride(arg837_1, (1024, ), (1, ))
    assert_size_stride(arg838_1, (), ())
    assert_size_stride(arg839_1, (416, ), (1, ))
    assert_size_stride(arg840_1, (416, ), (1, ))
    assert_size_stride(arg841_1, (), ())
    assert_size_stride(arg842_1, (104, ), (1, ))
    assert_size_stride(arg843_1, (104, ), (1, ))
    assert_size_stride(arg844_1, (), ())
    assert_size_stride(arg845_1, (104, ), (1, ))
    assert_size_stride(arg846_1, (104, ), (1, ))
    assert_size_stride(arg847_1, (), ())
    assert_size_stride(arg848_1, (104, ), (1, ))
    assert_size_stride(arg849_1, (104, ), (1, ))
    assert_size_stride(arg850_1, (), ())
    assert_size_stride(arg851_1, (1024, ), (1, ))
    assert_size_stride(arg852_1, (1024, ), (1, ))
    assert_size_stride(arg853_1, (), ())
    assert_size_stride(arg854_1, (416, ), (1, ))
    assert_size_stride(arg855_1, (416, ), (1, ))
    assert_size_stride(arg856_1, (), ())
    assert_size_stride(arg857_1, (104, ), (1, ))
    assert_size_stride(arg858_1, (104, ), (1, ))
    assert_size_stride(arg859_1, (), ())
    assert_size_stride(arg860_1, (104, ), (1, ))
    assert_size_stride(arg861_1, (104, ), (1, ))
    assert_size_stride(arg862_1, (), ())
    assert_size_stride(arg863_1, (104, ), (1, ))
    assert_size_stride(arg864_1, (104, ), (1, ))
    assert_size_stride(arg865_1, (), ())
    assert_size_stride(arg866_1, (1024, ), (1, ))
    assert_size_stride(arg867_1, (1024, ), (1, ))
    assert_size_stride(arg868_1, (), ())
    assert_size_stride(arg869_1, (416, ), (1, ))
    assert_size_stride(arg870_1, (416, ), (1, ))
    assert_size_stride(arg871_1, (), ())
    assert_size_stride(arg872_1, (104, ), (1, ))
    assert_size_stride(arg873_1, (104, ), (1, ))
    assert_size_stride(arg874_1, (), ())
    assert_size_stride(arg875_1, (104, ), (1, ))
    assert_size_stride(arg876_1, (104, ), (1, ))
    assert_size_stride(arg877_1, (), ())
    assert_size_stride(arg878_1, (104, ), (1, ))
    assert_size_stride(arg879_1, (104, ), (1, ))
    assert_size_stride(arg880_1, (), ())
    assert_size_stride(arg881_1, (1024, ), (1, ))
    assert_size_stride(arg882_1, (1024, ), (1, ))
    assert_size_stride(arg883_1, (), ())
    assert_size_stride(arg884_1, (416, ), (1, ))
    assert_size_stride(arg885_1, (416, ), (1, ))
    assert_size_stride(arg886_1, (), ())
    assert_size_stride(arg887_1, (104, ), (1, ))
    assert_size_stride(arg888_1, (104, ), (1, ))
    assert_size_stride(arg889_1, (), ())
    assert_size_stride(arg890_1, (104, ), (1, ))
    assert_size_stride(arg891_1, (104, ), (1, ))
    assert_size_stride(arg892_1, (), ())
    assert_size_stride(arg893_1, (104, ), (1, ))
    assert_size_stride(arg894_1, (104, ), (1, ))
    assert_size_stride(arg895_1, (), ())
    assert_size_stride(arg896_1, (1024, ), (1, ))
    assert_size_stride(arg897_1, (1024, ), (1, ))
    assert_size_stride(arg898_1, (), ())
    assert_size_stride(arg899_1, (416, ), (1, ))
    assert_size_stride(arg900_1, (416, ), (1, ))
    assert_size_stride(arg901_1, (), ())
    assert_size_stride(arg902_1, (104, ), (1, ))
    assert_size_stride(arg903_1, (104, ), (1, ))
    assert_size_stride(arg904_1, (), ())
    assert_size_stride(arg905_1, (104, ), (1, ))
    assert_size_stride(arg906_1, (104, ), (1, ))
    assert_size_stride(arg907_1, (), ())
    assert_size_stride(arg908_1, (104, ), (1, ))
    assert_size_stride(arg909_1, (104, ), (1, ))
    assert_size_stride(arg910_1, (), ())
    assert_size_stride(arg911_1, (1024, ), (1, ))
    assert_size_stride(arg912_1, (1024, ), (1, ))
    assert_size_stride(arg913_1, (), ())
    assert_size_stride(arg914_1, (416, ), (1, ))
    assert_size_stride(arg915_1, (416, ), (1, ))
    assert_size_stride(arg916_1, (), ())
    assert_size_stride(arg917_1, (104, ), (1, ))
    assert_size_stride(arg918_1, (104, ), (1, ))
    assert_size_stride(arg919_1, (), ())
    assert_size_stride(arg920_1, (104, ), (1, ))
    assert_size_stride(arg921_1, (104, ), (1, ))
    assert_size_stride(arg922_1, (), ())
    assert_size_stride(arg923_1, (104, ), (1, ))
    assert_size_stride(arg924_1, (104, ), (1, ))
    assert_size_stride(arg925_1, (), ())
    assert_size_stride(arg926_1, (1024, ), (1, ))
    assert_size_stride(arg927_1, (1024, ), (1, ))
    assert_size_stride(arg928_1, (), ())
    assert_size_stride(arg929_1, (416, ), (1, ))
    assert_size_stride(arg930_1, (416, ), (1, ))
    assert_size_stride(arg931_1, (), ())
    assert_size_stride(arg932_1, (104, ), (1, ))
    assert_size_stride(arg933_1, (104, ), (1, ))
    assert_size_stride(arg934_1, (), ())
    assert_size_stride(arg935_1, (104, ), (1, ))
    assert_size_stride(arg936_1, (104, ), (1, ))
    assert_size_stride(arg937_1, (), ())
    assert_size_stride(arg938_1, (104, ), (1, ))
    assert_size_stride(arg939_1, (104, ), (1, ))
    assert_size_stride(arg940_1, (), ())
    assert_size_stride(arg941_1, (1024, ), (1, ))
    assert_size_stride(arg942_1, (1024, ), (1, ))
    assert_size_stride(arg943_1, (), ())
    assert_size_stride(arg944_1, (416, ), (1, ))
    assert_size_stride(arg945_1, (416, ), (1, ))
    assert_size_stride(arg946_1, (), ())
    assert_size_stride(arg947_1, (104, ), (1, ))
    assert_size_stride(arg948_1, (104, ), (1, ))
    assert_size_stride(arg949_1, (), ())
    assert_size_stride(arg950_1, (104, ), (1, ))
    assert_size_stride(arg951_1, (104, ), (1, ))
    assert_size_stride(arg952_1, (), ())
    assert_size_stride(arg953_1, (104, ), (1, ))
    assert_size_stride(arg954_1, (104, ), (1, ))
    assert_size_stride(arg955_1, (), ())
    assert_size_stride(arg956_1, (1024, ), (1, ))
    assert_size_stride(arg957_1, (1024, ), (1, ))
    assert_size_stride(arg958_1, (), ())
    assert_size_stride(arg959_1, (416, ), (1, ))
    assert_size_stride(arg960_1, (416, ), (1, ))
    assert_size_stride(arg961_1, (), ())
    assert_size_stride(arg962_1, (104, ), (1, ))
    assert_size_stride(arg963_1, (104, ), (1, ))
    assert_size_stride(arg964_1, (), ())
    assert_size_stride(arg965_1, (104, ), (1, ))
    assert_size_stride(arg966_1, (104, ), (1, ))
    assert_size_stride(arg967_1, (), ())
    assert_size_stride(arg968_1, (104, ), (1, ))
    assert_size_stride(arg969_1, (104, ), (1, ))
    assert_size_stride(arg970_1, (), ())
    assert_size_stride(arg971_1, (1024, ), (1, ))
    assert_size_stride(arg972_1, (1024, ), (1, ))
    assert_size_stride(arg973_1, (), ())
    assert_size_stride(arg974_1, (832, ), (1, ))
    assert_size_stride(arg975_1, (832, ), (1, ))
    assert_size_stride(arg976_1, (), ())
    assert_size_stride(arg977_1, (208, ), (1, ))
    assert_size_stride(arg978_1, (208, ), (1, ))
    assert_size_stride(arg979_1, (), ())
    assert_size_stride(arg980_1, (208, ), (1, ))
    assert_size_stride(arg981_1, (208, ), (1, ))
    assert_size_stride(arg982_1, (), ())
    assert_size_stride(arg983_1, (208, ), (1, ))
    assert_size_stride(arg984_1, (208, ), (1, ))
    assert_size_stride(arg985_1, (), ())
    assert_size_stride(arg986_1, (2048, ), (1, ))
    assert_size_stride(arg987_1, (2048, ), (1, ))
    assert_size_stride(arg988_1, (), ())
    assert_size_stride(arg989_1, (2048, ), (1, ))
    assert_size_stride(arg990_1, (2048, ), (1, ))
    assert_size_stride(arg991_1, (), ())
    assert_size_stride(arg992_1, (832, ), (1, ))
    assert_size_stride(arg993_1, (832, ), (1, ))
    assert_size_stride(arg994_1, (), ())
    assert_size_stride(arg995_1, (208, ), (1, ))
    assert_size_stride(arg996_1, (208, ), (1, ))
    assert_size_stride(arg997_1, (), ())
    assert_size_stride(arg998_1, (208, ), (1, ))
    assert_size_stride(arg999_1, (208, ), (1, ))
    assert_size_stride(arg1000_1, (), ())
    assert_size_stride(arg1001_1, (208, ), (1, ))
    assert_size_stride(arg1002_1, (208, ), (1, ))
    assert_size_stride(arg1003_1, (), ())
    assert_size_stride(arg1004_1, (2048, ), (1, ))
    assert_size_stride(arg1005_1, (2048, ), (1, ))
    assert_size_stride(arg1006_1, (), ())
    assert_size_stride(arg1007_1, (832, ), (1, ))
    assert_size_stride(arg1008_1, (832, ), (1, ))
    assert_size_stride(arg1009_1, (), ())
    assert_size_stride(arg1010_1, (208, ), (1, ))
    assert_size_stride(arg1011_1, (208, ), (1, ))
    assert_size_stride(arg1012_1, (), ())
    assert_size_stride(arg1013_1, (208, ), (1, ))
    assert_size_stride(arg1014_1, (208, ), (1, ))
    assert_size_stride(arg1015_1, (), ())
    assert_size_stride(arg1016_1, (208, ), (1, ))
    assert_size_stride(arg1017_1, (208, ), (1, ))
    assert_size_stride(arg1018_1, (), ())
    assert_size_stride(arg1019_1, (2048, ), (1, ))
    assert_size_stride(arg1020_1, (2048, ), (1, ))
    assert_size_stride(arg1021_1, (), ())
    assert_size_stride(arg1022_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg1022_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1022_1
        buf1 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 49, grid=grid(192, 49), stream=stream0)
        del arg0_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg512_1, arg513_1, arg1_1, arg2_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg1_1
        del arg2_1
        del arg512_1
        del arg513_1
        buf4 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, buf4, 512, 3136, grid=grid(512, 3136), stream=stream0)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 104, 56, 56), (326144, 3136, 56, 1))
        del arg3_1
        buf6 = buf5; del buf5  # reuse
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf6, arg515_1, arg516_1, arg4_1, arg5_1, 2609152, grid=grid(2609152), stream=stream0)
        del arg4_1
        del arg515_1
        del arg516_1
        del arg5_1
        buf7 = empty_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf6, buf7, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf8 = empty_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg6_1, buf8, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg6_1
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf10 = buf7; del buf7  # reuse
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf6, buf10, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf11 = buf8; del buf8  # reuse
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg9_1, buf11, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg9_1
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf13 = buf10; del buf10  # reuse
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf6, buf13, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf14 = buf11; del buf11  # reuse
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg12_1, buf14, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg12_1
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf13, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 26, 56, 56), (81536, 3136, 56, 1))
        del buf13
        buf20 = empty((8, 104, 56, 56), device='cuda', dtype=torch.float32)
        buf16 = reinterpret_tensor(buf20, (8, 26, 56, 56), (326144, 3136, 56, 1), 244608)  # alias
        # Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_9.run(buf6, buf16, 652288, grid=grid(652288), stream=stream0)
        buf17 = reinterpret_tensor(buf20, (8, 26, 56, 56), (326144, 3136, 56, 1), 0)  # alias
        # Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf9, arg518_1, arg519_1, arg7_1, arg8_1, buf17, 652288, grid=grid(652288), stream=stream0)
        del arg518_1
        del arg519_1
        del arg7_1
        del arg8_1
        del buf9
        buf18 = reinterpret_tensor(buf20, (8, 26, 56, 56), (326144, 3136, 56, 1), 81536)  # alias
        # Source Nodes: [sp_6, sp_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf12, arg521_1, arg522_1, arg10_1, arg11_1, buf18, 652288, grid=grid(652288), stream=stream0)
        del arg10_1
        del arg11_1
        del arg521_1
        del arg522_1
        del buf12
        buf19 = reinterpret_tensor(buf20, (8, 26, 56, 56), (326144, 3136, 56, 1), 163072)  # alias
        # Source Nodes: [sp_10, sp_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf15, arg524_1, arg525_1, arg13_1, arg14_1, buf19, 652288, grid=grid(652288), stream=stream0)
        del arg13_1
        del arg14_1
        del arg524_1
        del arg525_1
        buf21 = reinterpret_tensor(buf6, (8, 104, 56, 56), (326144, 1, 5824, 104), 0); del buf6  # reuse
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(buf20, buf21, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf16
        del buf17
        del buf18
        del buf19
        del buf20
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg15_1
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf4, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg18_1
        buf24 = buf22; del buf22  # reuse
        buf25 = reinterpret_tensor(buf3, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf3  # reuse
        # Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf24, arg527_1, arg528_1, arg16_1, arg17_1, buf23, arg530_1, arg531_1, arg19_1, arg20_1, buf25, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        del arg16_1
        del arg17_1
        del arg19_1
        del arg20_1
        del arg527_1
        del arg528_1
        del arg530_1
        del arg531_1
        del buf23
        del buf24
        # Source Nodes: [out_8, shortcut_2], Original ATen: [aten.convolution, aten.relu]
        buf26 = extern_kernels.convolution(buf25, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 104, 56, 56), (326144, 3136, 56, 1))
        del arg21_1
        buf27 = buf26; del buf26  # reuse
        # Source Nodes: [out_10, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf27, arg533_1, arg534_1, arg22_1, arg23_1, 2609152, grid=grid(2609152), stream=stream0)
        del arg22_1
        del arg23_1
        del arg533_1
        del arg534_1
        buf28 = reinterpret_tensor(buf15, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf15  # reuse
        # Source Nodes: [sp_14], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf27, buf28, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf29 = buf14; del buf14  # reuse
        # Source Nodes: [sp_14], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg24_1, buf29, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg24_1
        # Source Nodes: [sp_14], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf41 = reinterpret_tensor(buf21, (8, 104, 56, 56), (326144, 3136, 56, 1), 0); del buf21  # reuse
        buf31 = reinterpret_tensor(buf41, (8, 26, 56, 56), (326144, 3136, 56, 1), 0)  # alias
        buf32 = buf28; del buf28  # reuse
        # Source Nodes: [sp_15, sp_16, sp_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf30, arg536_1, arg537_1, arg25_1, arg26_1, buf27, buf31, buf32, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg25_1
        del arg26_1
        del arg536_1
        del arg537_1
        del buf30
        buf33 = buf29; del buf29  # reuse
        # Source Nodes: [sp_17, sp_18], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg27_1, buf33, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg27_1
        # Source Nodes: [sp_17, sp_18], Original ATen: [aten.add, aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf35 = reinterpret_tensor(buf41, (8, 26, 56, 56), (326144, 3136, 56, 1), 81536)  # alias
        buf36 = buf32; del buf32  # reuse
        # Source Nodes: [sp_19, sp_20, sp_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf34, arg539_1, arg540_1, arg28_1, arg29_1, buf27, buf35, buf36, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg28_1
        del arg29_1
        del arg539_1
        del arg540_1
        del buf34
        buf37 = buf33; del buf33  # reuse
        # Source Nodes: [sp_21, sp_22], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg30_1, buf37, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg30_1
        # Source Nodes: [sp_21, sp_22], Original ATen: [aten.add, aten.convolution]
        buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 26, 56, 56), (81536, 3136, 56, 1))
        del buf36
        buf39 = reinterpret_tensor(buf41, (8, 26, 56, 56), (326144, 3136, 56, 1), 163072)  # alias
        # Source Nodes: [sp_23, sp_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf38, arg542_1, arg543_1, arg31_1, arg32_1, buf39, 652288, grid=grid(652288), stream=stream0)
        del arg31_1
        del arg32_1
        del arg542_1
        del arg543_1
        buf40 = reinterpret_tensor(buf41, (8, 26, 56, 56), (326144, 3136, 56, 1), 244608)  # alias
        # Source Nodes: [cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_15.run(buf27, buf40, 652288, grid=grid(652288), stream=stream0)
        buf42 = reinterpret_tensor(buf27, (8, 104, 56, 56), (326144, 1, 5824, 104), 0); del buf27  # reuse
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(buf41, buf42, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf31
        del buf35
        del buf39
        del buf40
        del buf41
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg33_1
        buf44 = buf25; del buf25  # reuse
        # Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf44, buf43, arg545_1, arg546_1, arg34_1, arg35_1, 25088, 256, grid=grid(25088, 256), stream=stream0)
        del arg34_1
        del arg35_1
        del arg545_1
        del arg546_1
        del buf43
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 104, 56, 56), (326144, 3136, 56, 1))
        del arg36_1
        buf46 = buf45; del buf45  # reuse
        # Source Nodes: [out_17, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf46, arg548_1, arg549_1, arg37_1, arg38_1, 2609152, grid=grid(2609152), stream=stream0)
        del arg37_1
        del arg38_1
        del arg548_1
        del arg549_1
        buf47 = reinterpret_tensor(buf38, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf38  # reuse
        # Source Nodes: [sp_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf46, buf47, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf48 = buf37; del buf37  # reuse
        # Source Nodes: [sp_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(arg39_1, buf48, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg39_1
        # Source Nodes: [sp_27], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf47, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf60 = reinterpret_tensor(buf42, (8, 104, 56, 56), (326144, 3136, 56, 1), 0); del buf42  # reuse
        buf50 = reinterpret_tensor(buf60, (8, 26, 56, 56), (326144, 3136, 56, 1), 0)  # alias
        buf51 = buf47; del buf47  # reuse
        # Source Nodes: [sp_28, sp_29, sp_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf49, arg551_1, arg552_1, arg40_1, arg41_1, buf46, buf50, buf51, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg40_1
        del arg41_1
        del arg551_1
        del arg552_1
        del buf49
        buf52 = buf48; del buf48  # reuse
        # Source Nodes: [sp_30, sp_31], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg42_1, buf52, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg42_1
        # Source Nodes: [sp_30, sp_31], Original ATen: [aten.add, aten.convolution]
        buf53 = extern_kernels.convolution(buf51, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (8, 26, 56, 56), (81536, 3136, 56, 1))
        buf54 = reinterpret_tensor(buf60, (8, 26, 56, 56), (326144, 3136, 56, 1), 81536)  # alias
        buf55 = buf51; del buf51  # reuse
        # Source Nodes: [sp_32, sp_33, sp_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf53, arg554_1, arg555_1, arg43_1, arg44_1, buf46, buf54, buf55, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg43_1
        del arg44_1
        del arg554_1
        del arg555_1
        del buf53
        buf56 = buf52; del buf52  # reuse
        # Source Nodes: [sp_34, sp_35], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_6.run(arg45_1, buf56, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg45_1
        # Source Nodes: [sp_34, sp_35], Original ATen: [aten.add, aten.convolution]
        buf57 = extern_kernels.convolution(buf55, buf56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 26, 56, 56), (81536, 3136, 56, 1))
        del buf56
        buf58 = reinterpret_tensor(buf60, (8, 26, 56, 56), (326144, 3136, 56, 1), 163072)  # alias
        # Source Nodes: [sp_36, sp_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf57, arg557_1, arg558_1, arg46_1, arg47_1, buf58, 652288, grid=grid(652288), stream=stream0)
        del arg46_1
        del arg47_1
        del arg557_1
        del arg558_1
        buf59 = reinterpret_tensor(buf60, (8, 26, 56, 56), (326144, 3136, 56, 1), 244608)  # alias
        # Source Nodes: [cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_15.run(buf46, buf59, 652288, grid=grid(652288), stream=stream0)
        buf61 = reinterpret_tensor(buf46, (8, 104, 56, 56), (326144, 1, 5824, 104), 0); del buf46  # reuse
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(buf60, buf61, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf50
        del buf54
        del buf58
        del buf59
        del buf60
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg48_1
        del buf61
        buf63 = buf44; del buf44  # reuse
        # Source Nodes: [out_21, out_22, shortcut_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf63, buf62, arg560_1, arg561_1, arg49_1, arg50_1, 25088, 256, grid=grid(25088, 256), stream=stream0)
        del arg49_1
        del arg50_1
        del arg560_1
        del arg561_1
        del buf62
        # Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 208, 56, 56), (652288, 3136, 56, 1))
        del arg51_1
        buf65 = buf64; del buf64  # reuse
        # Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf65, arg563_1, arg564_1, arg52_1, arg53_1, 5218304, grid=grid(5218304), stream=stream0)
        del arg52_1
        del arg53_1
        del arg563_1
        del arg564_1
        buf66 = empty_strided((8, 52, 56, 56), (163072, 1, 2912, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_40], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf65, buf66, 416, 3136, grid=grid(416, 3136), stream=stream0)
        buf67 = empty_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_40], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(arg54_1, buf67, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg54_1
        # Source Nodes: [sp_40], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf66, buf67, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf69 = buf66; del buf66  # reuse
        # Source Nodes: [sp_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf65, buf69, 416, 3136, grid=grid(416, 3136), stream=stream0)
        buf70 = buf67; del buf67  # reuse
        # Source Nodes: [sp_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(arg57_1, buf70, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg57_1
        # Source Nodes: [sp_44], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf69, buf70, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf72 = buf69; del buf69  # reuse
        # Source Nodes: [sp_48], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_21.run(buf65, buf72, 416, 3136, grid=grid(416, 3136), stream=stream0)
        buf73 = buf70; del buf70  # reuse
        # Source Nodes: [sp_48], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(arg60_1, buf73, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg60_1
        # Source Nodes: [sp_48], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf72, buf73, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf79 = reinterpret_tensor(buf72, (8, 208, 28, 28), (163072, 784, 28, 1), 0); del buf72  # reuse
        buf75 = reinterpret_tensor(buf79, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_22.run(buf65, buf75, 326144, grid=grid(326144), stream=stream0)
        del buf65
        buf76 = reinterpret_tensor(buf79, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        # Source Nodes: [sp_41, sp_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf68, arg566_1, arg567_1, arg55_1, arg56_1, buf76, 326144, grid=grid(326144), stream=stream0)
        del arg55_1
        del arg566_1
        del arg567_1
        del arg56_1
        del buf68
        buf77 = reinterpret_tensor(buf79, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        # Source Nodes: [sp_45, sp_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf71, arg569_1, arg570_1, arg58_1, arg59_1, buf77, 326144, grid=grid(326144), stream=stream0)
        del arg569_1
        del arg570_1
        del arg58_1
        del arg59_1
        del buf71
        buf78 = reinterpret_tensor(buf79, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        # Source Nodes: [sp_49, sp_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf74, arg572_1, arg573_1, arg61_1, arg62_1, buf78, 326144, grid=grid(326144), stream=stream0)
        del arg572_1
        del arg573_1
        del arg61_1
        del arg62_1
        buf80 = empty_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf79, buf80, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf75
        del buf76
        del buf77
        del buf78
        del buf79
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg63_1
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf63, arg66_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg66_1
        del buf63
        buf83 = buf81; del buf81  # reuse
        buf84 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf83, arg575_1, arg576_1, arg64_1, arg65_1, buf82, arg578_1, arg579_1, arg67_1, arg68_1, buf84, 4096, 784, grid=grid(4096, 784), stream=stream0)
        del arg575_1
        del arg576_1
        del arg578_1
        del arg579_1
        del arg64_1
        del arg65_1
        del arg67_1
        del arg68_1
        del buf82
        del buf83
        # Source Nodes: [out_32, shortcut_6], Original ATen: [aten.convolution, aten.relu]
        buf85 = extern_kernels.convolution(buf84, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 208, 28, 28), (163072, 784, 28, 1))
        del arg69_1
        buf86 = buf85; del buf85  # reuse
        # Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf86, arg581_1, arg582_1, arg70_1, arg71_1, 1304576, grid=grid(1304576), stream=stream0)
        del arg581_1
        del arg582_1
        del arg70_1
        del arg71_1
        buf87 = reinterpret_tensor(buf74, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf74  # reuse
        # Source Nodes: [sp_53], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf86, buf87, 416, 784, grid=grid(416, 784), stream=stream0)
        buf88 = buf73; del buf73  # reuse
        # Source Nodes: [sp_53], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(arg72_1, buf88, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg72_1
        # Source Nodes: [sp_53], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf87, buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf100 = reinterpret_tensor(buf80, (8, 208, 28, 28), (163072, 784, 28, 1), 0); del buf80  # reuse
        buf90 = reinterpret_tensor(buf100, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        buf91 = buf87; del buf87  # reuse
        # Source Nodes: [sp_54, sp_55, sp_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf89, arg584_1, arg585_1, arg73_1, arg74_1, buf86, buf90, buf91, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg584_1
        del arg585_1
        del arg73_1
        del arg74_1
        del buf89
        buf92 = buf88; del buf88  # reuse
        # Source Nodes: [sp_56, sp_57], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_19.run(arg75_1, buf92, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg75_1
        # Source Nodes: [sp_56, sp_57], Original ATen: [aten.add, aten.convolution]
        buf93 = extern_kernels.convolution(buf91, buf92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf94 = reinterpret_tensor(buf100, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        buf95 = buf91; del buf91  # reuse
        # Source Nodes: [sp_58, sp_59, sp_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29.run(buf93, arg587_1, arg588_1, arg76_1, arg77_1, buf86, buf94, buf95, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg587_1
        del arg588_1
        del arg76_1
        del arg77_1
        del buf93
        buf96 = buf92; del buf92  # reuse
        # Source Nodes: [sp_60, sp_61], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_19.run(arg78_1, buf96, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg78_1
        # Source Nodes: [sp_60, sp_61], Original ATen: [aten.add, aten.convolution]
        buf97 = extern_kernels.convolution(buf95, buf96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 52, 28, 28), (40768, 784, 28, 1))
        del buf95
        buf98 = reinterpret_tensor(buf100, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        # Source Nodes: [sp_62, sp_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf97, arg590_1, arg591_1, arg79_1, arg80_1, buf98, 326144, grid=grid(326144), stream=stream0)
        del arg590_1
        del arg591_1
        del arg79_1
        del arg80_1
        buf99 = reinterpret_tensor(buf100, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Source Nodes: [cat_61], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf86, buf99, 326144, grid=grid(326144), stream=stream0)
        buf101 = reinterpret_tensor(buf86, (8, 208, 28, 28), (163072, 1, 5824, 208), 0); del buf86  # reuse
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf100, buf101, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf100
        del buf90
        del buf94
        del buf98
        del buf99
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg81_1
        buf103 = buf84; del buf84  # reuse
        # Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31.run(buf103, buf102, arg593_1, arg594_1, arg82_1, arg83_1, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del arg593_1
        del arg594_1
        del arg82_1
        del arg83_1
        del buf102
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 208, 28, 28), (163072, 784, 28, 1))
        del arg84_1
        buf105 = buf104; del buf104  # reuse
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf105, arg596_1, arg597_1, arg85_1, arg86_1, 1304576, grid=grid(1304576), stream=stream0)
        del arg596_1
        del arg597_1
        del arg85_1
        del arg86_1
        buf106 = reinterpret_tensor(buf97, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf97  # reuse
        # Source Nodes: [sp_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf105, buf106, 416, 784, grid=grid(416, 784), stream=stream0)
        buf107 = buf96; del buf96  # reuse
        # Source Nodes: [sp_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(arg87_1, buf107, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg87_1
        # Source Nodes: [sp_66], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf106, buf107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf119 = reinterpret_tensor(buf101, (8, 208, 28, 28), (163072, 784, 28, 1), 0); del buf101  # reuse
        buf109 = reinterpret_tensor(buf119, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        buf110 = buf106; del buf106  # reuse
        # Source Nodes: [sp_67, sp_68, sp_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf108, arg599_1, arg600_1, arg88_1, arg89_1, buf105, buf109, buf110, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg599_1
        del arg600_1
        del arg88_1
        del arg89_1
        del buf108
        buf111 = buf107; del buf107  # reuse
        # Source Nodes: [sp_69, sp_70], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_19.run(arg90_1, buf111, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg90_1
        # Source Nodes: [sp_69, sp_70], Original ATen: [aten.add, aten.convolution]
        buf112 = extern_kernels.convolution(buf110, buf111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf113 = reinterpret_tensor(buf119, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        buf114 = buf110; del buf110  # reuse
        # Source Nodes: [sp_71, sp_72, sp_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29.run(buf112, arg602_1, arg603_1, arg91_1, arg92_1, buf105, buf113, buf114, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg602_1
        del arg603_1
        del arg91_1
        del arg92_1
        del buf112
        buf115 = buf111; del buf111  # reuse
        # Source Nodes: [sp_73, sp_74], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_19.run(arg93_1, buf115, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg93_1
        # Source Nodes: [sp_73, sp_74], Original ATen: [aten.add, aten.convolution]
        buf116 = extern_kernels.convolution(buf114, buf115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 52, 28, 28), (40768, 784, 28, 1))
        del buf114
        buf117 = reinterpret_tensor(buf119, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        # Source Nodes: [sp_75, sp_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf116, arg605_1, arg606_1, arg94_1, arg95_1, buf117, 326144, grid=grid(326144), stream=stream0)
        del arg605_1
        del arg606_1
        del arg94_1
        del arg95_1
        buf118 = reinterpret_tensor(buf119, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Source Nodes: [cat_60], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf105, buf118, 326144, grid=grid(326144), stream=stream0)
        buf120 = reinterpret_tensor(buf105, (8, 208, 28, 28), (163072, 1, 5824, 208), 0); del buf105  # reuse
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf119, buf120, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf109
        del buf113
        del buf117
        del buf118
        del buf119
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg96_1
        buf122 = buf103; del buf103  # reuse
        # Source Nodes: [out_45, out_46, shortcut_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31.run(buf122, buf121, arg608_1, arg609_1, arg97_1, arg98_1, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del arg608_1
        del arg609_1
        del arg97_1
        del arg98_1
        del buf121
        # Source Nodes: [out_48], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (8, 208, 28, 28), (163072, 784, 28, 1))
        del arg99_1
        buf124 = buf123; del buf123  # reuse
        # Source Nodes: [out_49, out_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf124, arg611_1, arg612_1, arg100_1, arg101_1, 1304576, grid=grid(1304576), stream=stream0)
        del arg100_1
        del arg101_1
        del arg611_1
        del arg612_1
        buf125 = reinterpret_tensor(buf116, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf116  # reuse
        # Source Nodes: [sp_79], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf124, buf125, 416, 784, grid=grid(416, 784), stream=stream0)
        buf126 = buf115; del buf115  # reuse
        # Source Nodes: [sp_79], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(arg102_1, buf126, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg102_1
        # Source Nodes: [sp_79], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf125, buf126, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf138 = reinterpret_tensor(buf120, (8, 208, 28, 28), (163072, 784, 28, 1), 0); del buf120  # reuse
        buf128 = reinterpret_tensor(buf138, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        buf129 = buf125; del buf125  # reuse
        # Source Nodes: [sp_80, sp_81, sp_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf127, arg614_1, arg615_1, arg103_1, arg104_1, buf124, buf128, buf129, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg103_1
        del arg104_1
        del arg614_1
        del arg615_1
        del buf127
        buf130 = buf126; del buf126  # reuse
        # Source Nodes: [sp_82, sp_83], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_19.run(arg105_1, buf130, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg105_1
        # Source Nodes: [sp_82, sp_83], Original ATen: [aten.add, aten.convolution]
        buf131 = extern_kernels.convolution(buf129, buf130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 52, 28, 28), (40768, 784, 28, 1))
        buf132 = reinterpret_tensor(buf138, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        buf133 = buf129; del buf129  # reuse
        # Source Nodes: [sp_84, sp_85, sp_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29.run(buf131, arg617_1, arg618_1, arg106_1, arg107_1, buf124, buf132, buf133, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg106_1
        del arg107_1
        del arg617_1
        del arg618_1
        del buf131
        buf134 = buf130; del buf130  # reuse
        # Source Nodes: [sp_86, sp_87], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_19.run(arg108_1, buf134, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg108_1
        # Source Nodes: [sp_86, sp_87], Original ATen: [aten.add, aten.convolution]
        buf135 = extern_kernels.convolution(buf133, buf134, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (8, 52, 28, 28), (40768, 784, 28, 1))
        del buf134
        buf136 = reinterpret_tensor(buf138, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        # Source Nodes: [sp_88, sp_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf135, arg620_1, arg621_1, arg109_1, arg110_1, buf136, 326144, grid=grid(326144), stream=stream0)
        del arg109_1
        del arg110_1
        del arg620_1
        del arg621_1
        buf137 = reinterpret_tensor(buf138, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Source Nodes: [cat_59], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf124, buf137, 326144, grid=grid(326144), stream=stream0)
        buf139 = reinterpret_tensor(buf124, (8, 208, 28, 28), (163072, 1, 5824, 208), 0); del buf124  # reuse
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf138, buf139, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf128
        del buf132
        del buf136
        del buf137
        del buf138
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg111_1
        del buf139
        buf141 = buf122; del buf122  # reuse
        # Source Nodes: [out_53, out_54, shortcut_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31.run(buf141, buf140, arg623_1, arg624_1, arg112_1, arg113_1, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del arg112_1
        del arg113_1
        del arg623_1
        del arg624_1
        del buf140
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 416, 28, 28), (326144, 784, 28, 1))
        del arg114_1
        buf143 = buf142; del buf142  # reuse
        # Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf143, arg626_1, arg627_1, arg115_1, arg116_1, 2609152, grid=grid(2609152), stream=stream0)
        del arg115_1
        del arg116_1
        del arg626_1
        del arg627_1
        buf144 = reinterpret_tensor(buf57, (8, 104, 28, 28), (81536, 1, 2912, 104), 0); del buf57  # reuse
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf143, buf144, 832, 784, grid=grid(832, 784), stream=stream0)
        buf145 = empty_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg117_1, buf145, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg117_1
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf144, buf145, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf147 = buf144; del buf144  # reuse
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(buf143, buf147, 832, 784, grid=grid(832, 784), stream=stream0)
        buf148 = buf145; del buf145  # reuse
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg120_1, buf148, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg120_1
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf147, buf148, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf150 = buf147; del buf147  # reuse
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf143, buf150, 832, 784, grid=grid(832, 784), stream=stream0)
        buf151 = buf148; del buf148  # reuse
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg123_1, buf151, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg123_1
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf150, buf151, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf157 = reinterpret_tensor(buf150, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf150  # reuse
        buf153 = reinterpret_tensor(buf157, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_37.run(buf143, buf153, 163072, grid=grid(163072), stream=stream0)
        del buf143
        buf154 = reinterpret_tensor(buf157, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Source Nodes: [sp_93, sp_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf146, arg629_1, arg630_1, arg118_1, arg119_1, buf154, 163072, grid=grid(163072), stream=stream0)
        del arg118_1
        del arg119_1
        del arg629_1
        del arg630_1
        del buf146
        buf155 = reinterpret_tensor(buf157, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Source Nodes: [sp_97, sp_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf149, arg632_1, arg633_1, arg121_1, arg122_1, buf155, 163072, grid=grid(163072), stream=stream0)
        del arg121_1
        del arg122_1
        del arg632_1
        del arg633_1
        del buf149
        buf156 = reinterpret_tensor(buf157, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_101, sp_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf152, arg635_1, arg636_1, arg124_1, arg125_1, buf156, 163072, grid=grid(163072), stream=stream0)
        del arg124_1
        del arg125_1
        del arg635_1
        del arg636_1
        buf158 = reinterpret_tensor(buf55, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf55  # reuse
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf157, buf158, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf153
        del buf154
        del buf155
        del buf156
        del buf157
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg126_1
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf141, arg129_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg129_1
        del buf141
        buf161 = buf159; del buf159  # reuse
        buf162 = reinterpret_tensor(buf4, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf4  # reuse
        # Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40.run(buf161, arg638_1, arg639_1, arg127_1, arg128_1, buf160, arg641_1, arg642_1, arg130_1, arg131_1, buf162, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del arg127_1
        del arg128_1
        del arg130_1
        del arg131_1
        del arg638_1
        del arg639_1
        del arg641_1
        del arg642_1
        del buf160
        del buf161
        # Source Nodes: [out_64, shortcut_11], Original ATen: [aten.convolution, aten.relu]
        buf163 = extern_kernels.convolution(buf162, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg132_1
        buf164 = buf163; del buf163  # reuse
        # Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf164, arg644_1, arg645_1, arg133_1, arg134_1, 652288, grid=grid(652288), stream=stream0)
        del arg133_1
        del arg134_1
        del arg644_1
        del arg645_1
        buf165 = reinterpret_tensor(buf152, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf152  # reuse
        # Source Nodes: [sp_105], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf164, buf165, 832, 196, grid=grid(832, 196), stream=stream0)
        buf166 = buf151; del buf151  # reuse
        # Source Nodes: [sp_105], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg135_1, buf166, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg135_1
        # Source Nodes: [sp_105], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf165, buf166, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf178 = reinterpret_tensor(buf158, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf158  # reuse
        buf168 = reinterpret_tensor(buf178, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf169 = buf165; del buf165  # reuse
        # Source Nodes: [sp_106, sp_107, sp_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf167, arg647_1, arg648_1, arg136_1, arg137_1, buf164, buf168, buf169, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg136_1
        del arg137_1
        del arg647_1
        del arg648_1
        del buf167
        buf170 = buf166; del buf166  # reuse
        # Source Nodes: [sp_108, sp_109], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg138_1, buf170, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg138_1
        # Source Nodes: [sp_108, sp_109], Original ATen: [aten.add, aten.convolution]
        buf171 = extern_kernels.convolution(buf169, buf170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf172 = reinterpret_tensor(buf178, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf173 = buf169; del buf169  # reuse
        # Source Nodes: [sp_110, sp_111, sp_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf171, arg650_1, arg651_1, arg139_1, arg140_1, buf164, buf172, buf173, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg139_1
        del arg140_1
        del arg650_1
        del arg651_1
        del buf171
        buf174 = buf170; del buf170  # reuse
        # Source Nodes: [sp_112, sp_113], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg141_1, buf174, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg141_1
        # Source Nodes: [sp_112, sp_113], Original ATen: [aten.add, aten.convolution]
        buf175 = extern_kernels.convolution(buf173, buf174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf173
        buf176 = reinterpret_tensor(buf178, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_114, sp_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf175, arg653_1, arg654_1, arg142_1, arg143_1, buf176, 163072, grid=grid(163072), stream=stream0)
        del arg142_1
        del arg143_1
        del arg653_1
        del arg654_1
        buf177 = reinterpret_tensor(buf178, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_57], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf164, buf177, 163072, grid=grid(163072), stream=stream0)
        buf179 = reinterpret_tensor(buf164, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf164  # reuse
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf178, buf179, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf168
        del buf172
        del buf176
        del buf177
        del buf178
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg144_1
        buf181 = buf162; del buf162  # reuse
        # Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf181, buf180, arg656_1, arg657_1, arg145_1, arg146_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg145_1
        del arg146_1
        del arg656_1
        del arg657_1
        del buf180
        # Source Nodes: [out_72], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg147_1
        buf183 = buf182; del buf182  # reuse
        # Source Nodes: [out_73, out_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf183, arg659_1, arg660_1, arg148_1, arg149_1, 652288, grid=grid(652288), stream=stream0)
        del arg148_1
        del arg149_1
        del arg659_1
        del arg660_1
        buf184 = reinterpret_tensor(buf175, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf175  # reuse
        # Source Nodes: [sp_118], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf183, buf184, 832, 196, grid=grid(832, 196), stream=stream0)
        buf185 = buf174; del buf174  # reuse
        # Source Nodes: [sp_118], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg150_1, buf185, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg150_1
        # Source Nodes: [sp_118], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf184, buf185, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf197 = reinterpret_tensor(buf179, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf179  # reuse
        buf187 = reinterpret_tensor(buf197, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf188 = buf184; del buf184  # reuse
        # Source Nodes: [sp_119, sp_120, sp_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf186, arg662_1, arg663_1, arg151_1, arg152_1, buf183, buf187, buf188, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg151_1
        del arg152_1
        del arg662_1
        del arg663_1
        del buf186
        buf189 = buf185; del buf185  # reuse
        # Source Nodes: [sp_121, sp_122], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg153_1, buf189, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg153_1
        # Source Nodes: [sp_121, sp_122], Original ATen: [aten.add, aten.convolution]
        buf190 = extern_kernels.convolution(buf188, buf189, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf191 = reinterpret_tensor(buf197, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf192 = buf188; del buf188  # reuse
        # Source Nodes: [sp_123, sp_124, sp_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf190, arg665_1, arg666_1, arg154_1, arg155_1, buf183, buf191, buf192, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg154_1
        del arg155_1
        del arg665_1
        del arg666_1
        del buf190
        buf193 = buf189; del buf189  # reuse
        # Source Nodes: [sp_125, sp_126], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg156_1, buf193, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg156_1
        # Source Nodes: [sp_125, sp_126], Original ATen: [aten.add, aten.convolution]
        buf194 = extern_kernels.convolution(buf192, buf193, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf192
        buf195 = reinterpret_tensor(buf197, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_127, sp_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf194, arg668_1, arg669_1, arg157_1, arg158_1, buf195, 163072, grid=grid(163072), stream=stream0)
        del arg157_1
        del arg158_1
        del arg668_1
        del arg669_1
        buf196 = reinterpret_tensor(buf197, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_56], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf183, buf196, 163072, grid=grid(163072), stream=stream0)
        buf198 = reinterpret_tensor(buf183, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf183  # reuse
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf197, buf198, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf187
        del buf191
        del buf195
        del buf196
        del buf197
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg159_1
        buf200 = buf181; del buf181  # reuse
        # Source Nodes: [out_77, out_78, shortcut_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf200, buf199, arg671_1, arg672_1, arg160_1, arg161_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg160_1
        del arg161_1
        del arg671_1
        del arg672_1
        del buf199
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg162_1
        buf202 = buf201; del buf201  # reuse
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf202, arg674_1, arg675_1, arg163_1, arg164_1, 652288, grid=grid(652288), stream=stream0)
        del arg163_1
        del arg164_1
        del arg674_1
        del arg675_1
        buf203 = reinterpret_tensor(buf194, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf194  # reuse
        # Source Nodes: [sp_131], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf202, buf203, 832, 196, grid=grid(832, 196), stream=stream0)
        buf204 = buf193; del buf193  # reuse
        # Source Nodes: [sp_131], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg165_1, buf204, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg165_1
        # Source Nodes: [sp_131], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf203, buf204, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf216 = reinterpret_tensor(buf198, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf198  # reuse
        buf206 = reinterpret_tensor(buf216, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf207 = buf203; del buf203  # reuse
        # Source Nodes: [sp_132, sp_133, sp_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf205, arg677_1, arg678_1, arg166_1, arg167_1, buf202, buf206, buf207, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg166_1
        del arg167_1
        del arg677_1
        del arg678_1
        del buf205
        buf208 = buf204; del buf204  # reuse
        # Source Nodes: [sp_134, sp_135], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg168_1, buf208, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg168_1
        # Source Nodes: [sp_134, sp_135], Original ATen: [aten.add, aten.convolution]
        buf209 = extern_kernels.convolution(buf207, buf208, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf210 = reinterpret_tensor(buf216, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf211 = buf207; del buf207  # reuse
        # Source Nodes: [sp_136, sp_137, sp_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf209, arg680_1, arg681_1, arg169_1, arg170_1, buf202, buf210, buf211, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg169_1
        del arg170_1
        del arg680_1
        del arg681_1
        del buf209
        buf212 = buf208; del buf208  # reuse
        # Source Nodes: [sp_138, sp_139], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg171_1, buf212, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg171_1
        # Source Nodes: [sp_138, sp_139], Original ATen: [aten.add, aten.convolution]
        buf213 = extern_kernels.convolution(buf211, buf212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf211
        buf214 = reinterpret_tensor(buf216, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_140, sp_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf213, arg683_1, arg684_1, arg172_1, arg173_1, buf214, 163072, grid=grid(163072), stream=stream0)
        del arg172_1
        del arg173_1
        del arg683_1
        del arg684_1
        buf215 = reinterpret_tensor(buf216, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_55], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf202, buf215, 163072, grid=grid(163072), stream=stream0)
        buf217 = reinterpret_tensor(buf202, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf202  # reuse
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf216, buf217, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf206
        del buf210
        del buf214
        del buf215
        del buf216
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg174_1
        buf219 = buf200; del buf200  # reuse
        # Source Nodes: [out_85, out_86, shortcut_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf219, buf218, arg686_1, arg687_1, arg175_1, arg176_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg175_1
        del arg176_1
        del arg686_1
        del arg687_1
        del buf218
        # Source Nodes: [out_88], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg177_1
        buf221 = buf220; del buf220  # reuse
        # Source Nodes: [out_89, out_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf221, arg689_1, arg690_1, arg178_1, arg179_1, 652288, grid=grid(652288), stream=stream0)
        del arg178_1
        del arg179_1
        del arg689_1
        del arg690_1
        buf222 = reinterpret_tensor(buf213, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf213  # reuse
        # Source Nodes: [sp_144], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf221, buf222, 832, 196, grid=grid(832, 196), stream=stream0)
        buf223 = buf212; del buf212  # reuse
        # Source Nodes: [sp_144], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg180_1, buf223, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg180_1
        # Source Nodes: [sp_144], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf222, buf223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf235 = reinterpret_tensor(buf217, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf217  # reuse
        buf225 = reinterpret_tensor(buf235, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf226 = buf222; del buf222  # reuse
        # Source Nodes: [sp_145, sp_146, sp_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf224, arg692_1, arg693_1, arg181_1, arg182_1, buf221, buf225, buf226, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg181_1
        del arg182_1
        del arg692_1
        del arg693_1
        del buf224
        buf227 = buf223; del buf223  # reuse
        # Source Nodes: [sp_147, sp_148], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg183_1, buf227, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg183_1
        # Source Nodes: [sp_147, sp_148], Original ATen: [aten.add, aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf229 = reinterpret_tensor(buf235, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf230 = buf226; del buf226  # reuse
        # Source Nodes: [sp_149, sp_150, sp_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf228, arg695_1, arg696_1, arg184_1, arg185_1, buf221, buf229, buf230, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg184_1
        del arg185_1
        del arg695_1
        del arg696_1
        del buf228
        buf231 = buf227; del buf227  # reuse
        # Source Nodes: [sp_151, sp_152], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg186_1, buf231, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg186_1
        # Source Nodes: [sp_151, sp_152], Original ATen: [aten.add, aten.convolution]
        buf232 = extern_kernels.convolution(buf230, buf231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf230
        buf233 = reinterpret_tensor(buf235, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_153, sp_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf232, arg698_1, arg699_1, arg187_1, arg188_1, buf233, 163072, grid=grid(163072), stream=stream0)
        del arg187_1
        del arg188_1
        del arg698_1
        del arg699_1
        buf234 = reinterpret_tensor(buf235, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_54], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf221, buf234, 163072, grid=grid(163072), stream=stream0)
        buf236 = reinterpret_tensor(buf221, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf221  # reuse
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf235, buf236, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf225
        del buf229
        del buf233
        del buf234
        del buf235
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg189_1
        buf238 = buf219; del buf219  # reuse
        # Source Nodes: [out_93, out_94, shortcut_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf238, buf237, arg701_1, arg702_1, arg190_1, arg191_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg190_1
        del arg191_1
        del arg701_1
        del arg702_1
        del buf237
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg192_1
        buf240 = buf239; del buf239  # reuse
        # Source Nodes: [out_97, out_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf240, arg704_1, arg705_1, arg193_1, arg194_1, 652288, grid=grid(652288), stream=stream0)
        del arg193_1
        del arg194_1
        del arg704_1
        del arg705_1
        buf241 = reinterpret_tensor(buf232, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf232  # reuse
        # Source Nodes: [sp_157], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf240, buf241, 832, 196, grid=grid(832, 196), stream=stream0)
        buf242 = buf231; del buf231  # reuse
        # Source Nodes: [sp_157], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg195_1, buf242, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg195_1
        # Source Nodes: [sp_157], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf241, buf242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf254 = reinterpret_tensor(buf236, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf236  # reuse
        buf244 = reinterpret_tensor(buf254, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf245 = buf241; del buf241  # reuse
        # Source Nodes: [sp_158, sp_159, sp_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf243, arg707_1, arg708_1, arg196_1, arg197_1, buf240, buf244, buf245, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg196_1
        del arg197_1
        del arg707_1
        del arg708_1
        del buf243
        buf246 = buf242; del buf242  # reuse
        # Source Nodes: [sp_160, sp_161], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg198_1, buf246, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg198_1
        # Source Nodes: [sp_160, sp_161], Original ATen: [aten.add, aten.convolution]
        buf247 = extern_kernels.convolution(buf245, buf246, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf248 = reinterpret_tensor(buf254, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf249 = buf245; del buf245  # reuse
        # Source Nodes: [sp_162, sp_163, sp_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf247, arg710_1, arg711_1, arg199_1, arg200_1, buf240, buf248, buf249, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg199_1
        del arg200_1
        del arg710_1
        del arg711_1
        del buf247
        buf250 = buf246; del buf246  # reuse
        # Source Nodes: [sp_164, sp_165], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg201_1, buf250, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg201_1
        # Source Nodes: [sp_164, sp_165], Original ATen: [aten.add, aten.convolution]
        buf251 = extern_kernels.convolution(buf249, buf250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf249
        buf252 = reinterpret_tensor(buf254, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_166, sp_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf251, arg713_1, arg714_1, arg202_1, arg203_1, buf252, 163072, grid=grid(163072), stream=stream0)
        del arg202_1
        del arg203_1
        del arg713_1
        del arg714_1
        buf253 = reinterpret_tensor(buf254, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_53], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf240, buf253, 163072, grid=grid(163072), stream=stream0)
        buf255 = reinterpret_tensor(buf240, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf240  # reuse
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf254, buf255, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf244
        del buf248
        del buf252
        del buf253
        del buf254
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg204_1
        buf257 = buf238; del buf238  # reuse
        # Source Nodes: [out_101, out_102, shortcut_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf257, buf256, arg716_1, arg717_1, arg205_1, arg206_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg205_1
        del arg206_1
        del arg716_1
        del arg717_1
        del buf256
        # Source Nodes: [out_104], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg207_1
        buf259 = buf258; del buf258  # reuse
        # Source Nodes: [out_105, out_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf259, arg719_1, arg720_1, arg208_1, arg209_1, 652288, grid=grid(652288), stream=stream0)
        del arg208_1
        del arg209_1
        del arg719_1
        del arg720_1
        buf260 = reinterpret_tensor(buf251, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf251  # reuse
        # Source Nodes: [sp_170], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf259, buf260, 832, 196, grid=grid(832, 196), stream=stream0)
        buf261 = buf250; del buf250  # reuse
        # Source Nodes: [sp_170], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg210_1, buf261, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg210_1
        # Source Nodes: [sp_170], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf260, buf261, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf273 = reinterpret_tensor(buf255, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf255  # reuse
        buf263 = reinterpret_tensor(buf273, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf264 = buf260; del buf260  # reuse
        # Source Nodes: [sp_171, sp_172, sp_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf262, arg722_1, arg723_1, arg211_1, arg212_1, buf259, buf263, buf264, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg211_1
        del arg212_1
        del arg722_1
        del arg723_1
        del buf262
        buf265 = buf261; del buf261  # reuse
        # Source Nodes: [sp_173, sp_174], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg213_1, buf265, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg213_1
        # Source Nodes: [sp_173, sp_174], Original ATen: [aten.add, aten.convolution]
        buf266 = extern_kernels.convolution(buf264, buf265, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf267 = reinterpret_tensor(buf273, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf268 = buf264; del buf264  # reuse
        # Source Nodes: [sp_175, sp_176, sp_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf266, arg725_1, arg726_1, arg214_1, arg215_1, buf259, buf267, buf268, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg214_1
        del arg215_1
        del arg725_1
        del arg726_1
        del buf266
        buf269 = buf265; del buf265  # reuse
        # Source Nodes: [sp_177, sp_178], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg216_1, buf269, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg216_1
        # Source Nodes: [sp_177, sp_178], Original ATen: [aten.add, aten.convolution]
        buf270 = extern_kernels.convolution(buf268, buf269, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf268
        buf271 = reinterpret_tensor(buf273, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_179, sp_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf270, arg728_1, arg729_1, arg217_1, arg218_1, buf271, 163072, grid=grid(163072), stream=stream0)
        del arg217_1
        del arg218_1
        del arg728_1
        del arg729_1
        buf272 = reinterpret_tensor(buf273, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_52], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf259, buf272, 163072, grid=grid(163072), stream=stream0)
        buf274 = reinterpret_tensor(buf259, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf259  # reuse
        # Source Nodes: [out_108], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf273, buf274, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf263
        del buf267
        del buf271
        del buf272
        del buf273
        # Source Nodes: [out_108], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg219_1
        buf276 = buf257; del buf257  # reuse
        # Source Nodes: [out_109, out_110, shortcut_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf276, buf275, arg731_1, arg732_1, arg220_1, arg221_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg220_1
        del arg221_1
        del arg731_1
        del arg732_1
        del buf275
        # Source Nodes: [out_112], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg222_1
        buf278 = buf277; del buf277  # reuse
        # Source Nodes: [out_113, out_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf278, arg734_1, arg735_1, arg223_1, arg224_1, 652288, grid=grid(652288), stream=stream0)
        del arg223_1
        del arg224_1
        del arg734_1
        del arg735_1
        buf279 = reinterpret_tensor(buf270, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf270  # reuse
        # Source Nodes: [sp_183], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf278, buf279, 832, 196, grid=grid(832, 196), stream=stream0)
        buf280 = buf269; del buf269  # reuse
        # Source Nodes: [sp_183], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg225_1, buf280, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg225_1
        # Source Nodes: [sp_183], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf279, buf280, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf292 = reinterpret_tensor(buf274, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf274  # reuse
        buf282 = reinterpret_tensor(buf292, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf283 = buf279; del buf279  # reuse
        # Source Nodes: [sp_184, sp_185, sp_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf281, arg737_1, arg738_1, arg226_1, arg227_1, buf278, buf282, buf283, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg226_1
        del arg227_1
        del arg737_1
        del arg738_1
        del buf281
        buf284 = buf280; del buf280  # reuse
        # Source Nodes: [sp_186, sp_187], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg228_1, buf284, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg228_1
        # Source Nodes: [sp_186, sp_187], Original ATen: [aten.add, aten.convolution]
        buf285 = extern_kernels.convolution(buf283, buf284, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf286 = reinterpret_tensor(buf292, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf287 = buf283; del buf283  # reuse
        # Source Nodes: [sp_188, sp_189, sp_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf285, arg740_1, arg741_1, arg229_1, arg230_1, buf278, buf286, buf287, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg229_1
        del arg230_1
        del arg740_1
        del arg741_1
        del buf285
        buf288 = buf284; del buf284  # reuse
        # Source Nodes: [sp_190, sp_191], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg231_1, buf288, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg231_1
        # Source Nodes: [sp_190, sp_191], Original ATen: [aten.add, aten.convolution]
        buf289 = extern_kernels.convolution(buf287, buf288, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf287
        buf290 = reinterpret_tensor(buf292, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_192, sp_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf289, arg743_1, arg744_1, arg232_1, arg233_1, buf290, 163072, grid=grid(163072), stream=stream0)
        del arg232_1
        del arg233_1
        del arg743_1
        del arg744_1
        buf291 = reinterpret_tensor(buf292, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_51], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf278, buf291, 163072, grid=grid(163072), stream=stream0)
        buf293 = reinterpret_tensor(buf278, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf278  # reuse
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf292, buf293, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf282
        del buf286
        del buf290
        del buf291
        del buf292
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg234_1
        buf295 = buf276; del buf276  # reuse
        # Source Nodes: [out_117, out_118, shortcut_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf295, buf294, arg746_1, arg747_1, arg235_1, arg236_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg235_1
        del arg236_1
        del arg746_1
        del arg747_1
        del buf294
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg237_1
        buf297 = buf296; del buf296  # reuse
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf297, arg749_1, arg750_1, arg238_1, arg239_1, 652288, grid=grid(652288), stream=stream0)
        del arg238_1
        del arg239_1
        del arg749_1
        del arg750_1
        buf298 = reinterpret_tensor(buf289, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf289  # reuse
        # Source Nodes: [sp_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf297, buf298, 832, 196, grid=grid(832, 196), stream=stream0)
        buf299 = buf288; del buf288  # reuse
        # Source Nodes: [sp_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg240_1, buf299, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg240_1
        # Source Nodes: [sp_196], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf298, buf299, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf311 = reinterpret_tensor(buf293, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf293  # reuse
        buf301 = reinterpret_tensor(buf311, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf302 = buf298; del buf298  # reuse
        # Source Nodes: [sp_197, sp_198, sp_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf300, arg752_1, arg753_1, arg241_1, arg242_1, buf297, buf301, buf302, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg241_1
        del arg242_1
        del arg752_1
        del arg753_1
        del buf300
        buf303 = buf299; del buf299  # reuse
        # Source Nodes: [sp_199, sp_200], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg243_1, buf303, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg243_1
        # Source Nodes: [sp_199, sp_200], Original ATen: [aten.add, aten.convolution]
        buf304 = extern_kernels.convolution(buf302, buf303, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf305 = reinterpret_tensor(buf311, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf306 = buf302; del buf302  # reuse
        # Source Nodes: [sp_201, sp_202, sp_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf304, arg755_1, arg756_1, arg244_1, arg245_1, buf297, buf305, buf306, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg244_1
        del arg245_1
        del arg755_1
        del arg756_1
        del buf304
        buf307 = buf303; del buf303  # reuse
        # Source Nodes: [sp_203, sp_204], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg246_1, buf307, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg246_1
        # Source Nodes: [sp_203, sp_204], Original ATen: [aten.add, aten.convolution]
        buf308 = extern_kernels.convolution(buf306, buf307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf306
        buf309 = reinterpret_tensor(buf311, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_205, sp_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf308, arg758_1, arg759_1, arg247_1, arg248_1, buf309, 163072, grid=grid(163072), stream=stream0)
        del arg247_1
        del arg248_1
        del arg758_1
        del arg759_1
        buf310 = reinterpret_tensor(buf311, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_50], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf297, buf310, 163072, grid=grid(163072), stream=stream0)
        buf312 = reinterpret_tensor(buf297, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf297  # reuse
        # Source Nodes: [out_124], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf311, buf312, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf301
        del buf305
        del buf309
        del buf310
        del buf311
        # Source Nodes: [out_124], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg249_1
        buf314 = buf295; del buf295  # reuse
        # Source Nodes: [out_125, out_126, shortcut_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf314, buf313, arg761_1, arg762_1, arg250_1, arg251_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg250_1
        del arg251_1
        del arg761_1
        del arg762_1
        del buf313
        # Source Nodes: [out_128], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf314, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg252_1
        buf316 = buf315; del buf315  # reuse
        # Source Nodes: [out_129, out_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf316, arg764_1, arg765_1, arg253_1, arg254_1, 652288, grid=grid(652288), stream=stream0)
        del arg253_1
        del arg254_1
        del arg764_1
        del arg765_1
        buf317 = reinterpret_tensor(buf308, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf308  # reuse
        # Source Nodes: [sp_209], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf316, buf317, 832, 196, grid=grid(832, 196), stream=stream0)
        buf318 = buf307; del buf307  # reuse
        # Source Nodes: [sp_209], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg255_1, buf318, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg255_1
        # Source Nodes: [sp_209], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf317, buf318, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf330 = reinterpret_tensor(buf312, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf312  # reuse
        buf320 = reinterpret_tensor(buf330, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf321 = buf317; del buf317  # reuse
        # Source Nodes: [sp_210, sp_211, sp_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf319, arg767_1, arg768_1, arg256_1, arg257_1, buf316, buf320, buf321, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg256_1
        del arg257_1
        del arg767_1
        del arg768_1
        del buf319
        buf322 = buf318; del buf318  # reuse
        # Source Nodes: [sp_212, sp_213], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg258_1, buf322, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg258_1
        # Source Nodes: [sp_212, sp_213], Original ATen: [aten.add, aten.convolution]
        buf323 = extern_kernels.convolution(buf321, buf322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf324 = reinterpret_tensor(buf330, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf325 = buf321; del buf321  # reuse
        # Source Nodes: [sp_214, sp_215, sp_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf323, arg770_1, arg771_1, arg259_1, arg260_1, buf316, buf324, buf325, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg259_1
        del arg260_1
        del arg770_1
        del arg771_1
        del buf323
        buf326 = buf322; del buf322  # reuse
        # Source Nodes: [sp_216, sp_217], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg261_1, buf326, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg261_1
        # Source Nodes: [sp_216, sp_217], Original ATen: [aten.add, aten.convolution]
        buf327 = extern_kernels.convolution(buf325, buf326, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf325
        buf328 = reinterpret_tensor(buf330, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_218, sp_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf327, arg773_1, arg774_1, arg262_1, arg263_1, buf328, 163072, grid=grid(163072), stream=stream0)
        del arg262_1
        del arg263_1
        del arg773_1
        del arg774_1
        buf329 = reinterpret_tensor(buf330, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_49], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf316, buf329, 163072, grid=grid(163072), stream=stream0)
        buf331 = reinterpret_tensor(buf316, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf316  # reuse
        # Source Nodes: [out_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf330, buf331, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf320
        del buf324
        del buf328
        del buf329
        del buf330
        # Source Nodes: [out_132], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg264_1
        buf333 = buf314; del buf314  # reuse
        # Source Nodes: [out_133, out_134, shortcut_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf333, buf332, arg776_1, arg777_1, arg265_1, arg266_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg265_1
        del arg266_1
        del arg776_1
        del arg777_1
        del buf332
        # Source Nodes: [out_136], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, arg267_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg267_1
        buf335 = buf334; del buf334  # reuse
        # Source Nodes: [out_137, out_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf335, arg779_1, arg780_1, arg268_1, arg269_1, 652288, grid=grid(652288), stream=stream0)
        del arg268_1
        del arg269_1
        del arg779_1
        del arg780_1
        buf336 = reinterpret_tensor(buf327, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf327  # reuse
        # Source Nodes: [sp_222], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf335, buf336, 832, 196, grid=grid(832, 196), stream=stream0)
        buf337 = buf326; del buf326  # reuse
        # Source Nodes: [sp_222], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg270_1, buf337, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg270_1
        # Source Nodes: [sp_222], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf336, buf337, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf349 = reinterpret_tensor(buf331, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf331  # reuse
        buf339 = reinterpret_tensor(buf349, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf340 = buf336; del buf336  # reuse
        # Source Nodes: [sp_223, sp_224, sp_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf338, arg782_1, arg783_1, arg271_1, arg272_1, buf335, buf339, buf340, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg271_1
        del arg272_1
        del arg782_1
        del arg783_1
        del buf338
        buf341 = buf337; del buf337  # reuse
        # Source Nodes: [sp_225, sp_226], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg273_1, buf341, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg273_1
        # Source Nodes: [sp_225, sp_226], Original ATen: [aten.add, aten.convolution]
        buf342 = extern_kernels.convolution(buf340, buf341, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf343 = reinterpret_tensor(buf349, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf344 = buf340; del buf340  # reuse
        # Source Nodes: [sp_227, sp_228, sp_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf342, arg785_1, arg786_1, arg274_1, arg275_1, buf335, buf343, buf344, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg274_1
        del arg275_1
        del arg785_1
        del arg786_1
        del buf342
        buf345 = buf341; del buf341  # reuse
        # Source Nodes: [sp_229, sp_230], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg276_1, buf345, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg276_1
        # Source Nodes: [sp_229, sp_230], Original ATen: [aten.add, aten.convolution]
        buf346 = extern_kernels.convolution(buf344, buf345, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf344
        buf347 = reinterpret_tensor(buf349, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_231, sp_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf346, arg788_1, arg789_1, arg277_1, arg278_1, buf347, 163072, grid=grid(163072), stream=stream0)
        del arg277_1
        del arg278_1
        del arg788_1
        del arg789_1
        buf348 = reinterpret_tensor(buf349, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_48], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf335, buf348, 163072, grid=grid(163072), stream=stream0)
        buf350 = reinterpret_tensor(buf335, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf335  # reuse
        # Source Nodes: [out_140], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf349, buf350, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf339
        del buf343
        del buf347
        del buf348
        del buf349
        # Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg279_1
        buf352 = buf333; del buf333  # reuse
        # Source Nodes: [out_141, out_142, shortcut_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf352, buf351, arg791_1, arg792_1, arg280_1, arg281_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg280_1
        del arg281_1
        del arg791_1
        del arg792_1
        del buf351
        # Source Nodes: [out_144], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg282_1
        buf354 = buf353; del buf353  # reuse
        # Source Nodes: [out_145, out_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf354, arg794_1, arg795_1, arg283_1, arg284_1, 652288, grid=grid(652288), stream=stream0)
        del arg283_1
        del arg284_1
        del arg794_1
        del arg795_1
        buf355 = reinterpret_tensor(buf346, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf346  # reuse
        # Source Nodes: [sp_235], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf354, buf355, 832, 196, grid=grid(832, 196), stream=stream0)
        buf356 = buf345; del buf345  # reuse
        # Source Nodes: [sp_235], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg285_1, buf356, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg285_1
        # Source Nodes: [sp_235], Original ATen: [aten.convolution]
        buf357 = extern_kernels.convolution(buf355, buf356, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf357, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf368 = reinterpret_tensor(buf350, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf350  # reuse
        buf358 = reinterpret_tensor(buf368, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf359 = buf355; del buf355  # reuse
        # Source Nodes: [sp_236, sp_237, sp_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf357, arg797_1, arg798_1, arg286_1, arg287_1, buf354, buf358, buf359, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg286_1
        del arg287_1
        del arg797_1
        del arg798_1
        del buf357
        buf360 = buf356; del buf356  # reuse
        # Source Nodes: [sp_238, sp_239], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg288_1, buf360, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg288_1
        # Source Nodes: [sp_238, sp_239], Original ATen: [aten.add, aten.convolution]
        buf361 = extern_kernels.convolution(buf359, buf360, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf362 = reinterpret_tensor(buf368, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf363 = buf359; del buf359  # reuse
        # Source Nodes: [sp_240, sp_241, sp_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf361, arg800_1, arg801_1, arg289_1, arg290_1, buf354, buf362, buf363, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg289_1
        del arg290_1
        del arg800_1
        del arg801_1
        del buf361
        buf364 = buf360; del buf360  # reuse
        # Source Nodes: [sp_242, sp_243], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg291_1, buf364, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg291_1
        # Source Nodes: [sp_242, sp_243], Original ATen: [aten.add, aten.convolution]
        buf365 = extern_kernels.convolution(buf363, buf364, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf363
        buf366 = reinterpret_tensor(buf368, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_244, sp_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf365, arg803_1, arg804_1, arg292_1, arg293_1, buf366, 163072, grid=grid(163072), stream=stream0)
        del arg292_1
        del arg293_1
        del arg803_1
        del arg804_1
        buf367 = reinterpret_tensor(buf368, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_47], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf354, buf367, 163072, grid=grid(163072), stream=stream0)
        buf369 = reinterpret_tensor(buf354, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf354  # reuse
        # Source Nodes: [out_148], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf368, buf369, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf358
        del buf362
        del buf366
        del buf367
        del buf368
        # Source Nodes: [out_148], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(buf369, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg294_1
        buf371 = buf352; del buf352  # reuse
        # Source Nodes: [out_149, out_150, shortcut_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf371, buf370, arg806_1, arg807_1, arg295_1, arg296_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg295_1
        del arg296_1
        del arg806_1
        del arg807_1
        del buf370
        # Source Nodes: [out_152], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, arg297_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg297_1
        buf373 = buf372; del buf372  # reuse
        # Source Nodes: [out_153, out_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf373, arg809_1, arg810_1, arg298_1, arg299_1, 652288, grid=grid(652288), stream=stream0)
        del arg298_1
        del arg299_1
        del arg809_1
        del arg810_1
        buf374 = reinterpret_tensor(buf365, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf365  # reuse
        # Source Nodes: [sp_248], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf373, buf374, 832, 196, grid=grid(832, 196), stream=stream0)
        buf375 = buf364; del buf364  # reuse
        # Source Nodes: [sp_248], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg300_1, buf375, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg300_1
        # Source Nodes: [sp_248], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf374, buf375, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf387 = reinterpret_tensor(buf369, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf369  # reuse
        buf377 = reinterpret_tensor(buf387, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf378 = buf374; del buf374  # reuse
        # Source Nodes: [sp_249, sp_250, sp_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf376, arg812_1, arg813_1, arg301_1, arg302_1, buf373, buf377, buf378, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg301_1
        del arg302_1
        del arg812_1
        del arg813_1
        del buf376
        buf379 = buf375; del buf375  # reuse
        # Source Nodes: [sp_251, sp_252], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg303_1, buf379, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg303_1
        # Source Nodes: [sp_251, sp_252], Original ATen: [aten.add, aten.convolution]
        buf380 = extern_kernels.convolution(buf378, buf379, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf381 = reinterpret_tensor(buf387, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf382 = buf378; del buf378  # reuse
        # Source Nodes: [sp_253, sp_254, sp_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf380, arg815_1, arg816_1, arg304_1, arg305_1, buf373, buf381, buf382, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg304_1
        del arg305_1
        del arg815_1
        del arg816_1
        del buf380
        buf383 = buf379; del buf379  # reuse
        # Source Nodes: [sp_255, sp_256], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg306_1, buf383, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg306_1
        # Source Nodes: [sp_255, sp_256], Original ATen: [aten.add, aten.convolution]
        buf384 = extern_kernels.convolution(buf382, buf383, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf384, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf382
        buf385 = reinterpret_tensor(buf387, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_257, sp_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf384, arg818_1, arg819_1, arg307_1, arg308_1, buf385, 163072, grid=grid(163072), stream=stream0)
        del arg307_1
        del arg308_1
        del arg818_1
        del arg819_1
        buf386 = reinterpret_tensor(buf387, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_46], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf373, buf386, 163072, grid=grid(163072), stream=stream0)
        buf388 = reinterpret_tensor(buf373, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf373  # reuse
        # Source Nodes: [out_156], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf387, buf388, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf377
        del buf381
        del buf385
        del buf386
        del buf387
        # Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, arg309_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg309_1
        buf390 = buf371; del buf371  # reuse
        # Source Nodes: [out_157, out_158, shortcut_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf390, buf389, arg821_1, arg822_1, arg310_1, arg311_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg310_1
        del arg311_1
        del arg821_1
        del arg822_1
        del buf389
        # Source Nodes: [out_160], Original ATen: [aten.convolution]
        buf391 = extern_kernels.convolution(buf390, arg312_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf391, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg312_1
        buf392 = buf391; del buf391  # reuse
        # Source Nodes: [out_161, out_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf392, arg824_1, arg825_1, arg313_1, arg314_1, 652288, grid=grid(652288), stream=stream0)
        del arg313_1
        del arg314_1
        del arg824_1
        del arg825_1
        buf393 = reinterpret_tensor(buf384, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf384  # reuse
        # Source Nodes: [sp_261], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf392, buf393, 832, 196, grid=grid(832, 196), stream=stream0)
        buf394 = buf383; del buf383  # reuse
        # Source Nodes: [sp_261], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg315_1, buf394, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg315_1
        # Source Nodes: [sp_261], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf393, buf394, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf406 = reinterpret_tensor(buf388, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf388  # reuse
        buf396 = reinterpret_tensor(buf406, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf397 = buf393; del buf393  # reuse
        # Source Nodes: [sp_262, sp_263, sp_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf395, arg827_1, arg828_1, arg316_1, arg317_1, buf392, buf396, buf397, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg316_1
        del arg317_1
        del arg827_1
        del arg828_1
        del buf395
        buf398 = buf394; del buf394  # reuse
        # Source Nodes: [sp_264, sp_265], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg318_1, buf398, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg318_1
        # Source Nodes: [sp_264, sp_265], Original ATen: [aten.add, aten.convolution]
        buf399 = extern_kernels.convolution(buf397, buf398, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf400 = reinterpret_tensor(buf406, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf401 = buf397; del buf397  # reuse
        # Source Nodes: [sp_266, sp_267, sp_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf399, arg830_1, arg831_1, arg319_1, arg320_1, buf392, buf400, buf401, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg319_1
        del arg320_1
        del arg830_1
        del arg831_1
        del buf399
        buf402 = buf398; del buf398  # reuse
        # Source Nodes: [sp_268, sp_269], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg321_1, buf402, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg321_1
        # Source Nodes: [sp_268, sp_269], Original ATen: [aten.add, aten.convolution]
        buf403 = extern_kernels.convolution(buf401, buf402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf401
        buf404 = reinterpret_tensor(buf406, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_270, sp_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf403, arg833_1, arg834_1, arg322_1, arg323_1, buf404, 163072, grid=grid(163072), stream=stream0)
        del arg322_1
        del arg323_1
        del arg833_1
        del arg834_1
        buf405 = reinterpret_tensor(buf406, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_45], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf392, buf405, 163072, grid=grid(163072), stream=stream0)
        buf407 = reinterpret_tensor(buf392, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf392  # reuse
        # Source Nodes: [out_164], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf406, buf407, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf396
        del buf400
        del buf404
        del buf405
        del buf406
        # Source Nodes: [out_164], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf407, arg324_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg324_1
        buf409 = buf390; del buf390  # reuse
        # Source Nodes: [out_165, out_166, shortcut_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf409, buf408, arg836_1, arg837_1, arg325_1, arg326_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg325_1
        del arg326_1
        del arg836_1
        del arg837_1
        del buf408
        # Source Nodes: [out_168], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf409, arg327_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg327_1
        buf411 = buf410; del buf410  # reuse
        # Source Nodes: [out_169, out_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf411, arg839_1, arg840_1, arg328_1, arg329_1, 652288, grid=grid(652288), stream=stream0)
        del arg328_1
        del arg329_1
        del arg839_1
        del arg840_1
        buf412 = reinterpret_tensor(buf403, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf403  # reuse
        # Source Nodes: [sp_274], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf411, buf412, 832, 196, grid=grid(832, 196), stream=stream0)
        buf413 = buf402; del buf402  # reuse
        # Source Nodes: [sp_274], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg330_1, buf413, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg330_1
        # Source Nodes: [sp_274], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf412, buf413, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf425 = reinterpret_tensor(buf407, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf407  # reuse
        buf415 = reinterpret_tensor(buf425, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf416 = buf412; del buf412  # reuse
        # Source Nodes: [sp_275, sp_276, sp_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf414, arg842_1, arg843_1, arg331_1, arg332_1, buf411, buf415, buf416, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg331_1
        del arg332_1
        del arg842_1
        del arg843_1
        del buf414
        buf417 = buf413; del buf413  # reuse
        # Source Nodes: [sp_277, sp_278], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg333_1, buf417, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg333_1
        # Source Nodes: [sp_277, sp_278], Original ATen: [aten.add, aten.convolution]
        buf418 = extern_kernels.convolution(buf416, buf417, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf419 = reinterpret_tensor(buf425, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf420 = buf416; del buf416  # reuse
        # Source Nodes: [sp_279, sp_280, sp_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf418, arg845_1, arg846_1, arg334_1, arg335_1, buf411, buf419, buf420, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg334_1
        del arg335_1
        del arg845_1
        del arg846_1
        del buf418
        buf421 = buf417; del buf417  # reuse
        # Source Nodes: [sp_281, sp_282], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg336_1, buf421, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg336_1
        # Source Nodes: [sp_281, sp_282], Original ATen: [aten.add, aten.convolution]
        buf422 = extern_kernels.convolution(buf420, buf421, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf420
        buf423 = reinterpret_tensor(buf425, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_283, sp_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf422, arg848_1, arg849_1, arg337_1, arg338_1, buf423, 163072, grid=grid(163072), stream=stream0)
        del arg337_1
        del arg338_1
        del arg848_1
        del arg849_1
        buf424 = reinterpret_tensor(buf425, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_44], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf411, buf424, 163072, grid=grid(163072), stream=stream0)
        buf426 = reinterpret_tensor(buf411, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf411  # reuse
        # Source Nodes: [out_172], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf425, buf426, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf415
        del buf419
        del buf423
        del buf424
        del buf425
        # Source Nodes: [out_172], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, arg339_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg339_1
        buf428 = buf409; del buf409  # reuse
        # Source Nodes: [out_173, out_174, shortcut_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf428, buf427, arg851_1, arg852_1, arg340_1, arg341_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg340_1
        del arg341_1
        del arg851_1
        del arg852_1
        del buf427
        # Source Nodes: [out_176], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf428, arg342_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg342_1
        buf430 = buf429; del buf429  # reuse
        # Source Nodes: [out_177, out_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf430, arg854_1, arg855_1, arg343_1, arg344_1, 652288, grid=grid(652288), stream=stream0)
        del arg343_1
        del arg344_1
        del arg854_1
        del arg855_1
        buf431 = reinterpret_tensor(buf422, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf422  # reuse
        # Source Nodes: [sp_287], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf430, buf431, 832, 196, grid=grid(832, 196), stream=stream0)
        buf432 = buf421; del buf421  # reuse
        # Source Nodes: [sp_287], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg345_1, buf432, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg345_1
        # Source Nodes: [sp_287], Original ATen: [aten.convolution]
        buf433 = extern_kernels.convolution(buf431, buf432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf433, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf444 = reinterpret_tensor(buf426, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf426  # reuse
        buf434 = reinterpret_tensor(buf444, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf435 = buf431; del buf431  # reuse
        # Source Nodes: [sp_288, sp_289, sp_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf433, arg857_1, arg858_1, arg346_1, arg347_1, buf430, buf434, buf435, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg346_1
        del arg347_1
        del arg857_1
        del arg858_1
        del buf433
        buf436 = buf432; del buf432  # reuse
        # Source Nodes: [sp_290, sp_291], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg348_1, buf436, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg348_1
        # Source Nodes: [sp_290, sp_291], Original ATen: [aten.add, aten.convolution]
        buf437 = extern_kernels.convolution(buf435, buf436, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf438 = reinterpret_tensor(buf444, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf439 = buf435; del buf435  # reuse
        # Source Nodes: [sp_292, sp_293, sp_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf437, arg860_1, arg861_1, arg349_1, arg350_1, buf430, buf438, buf439, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg349_1
        del arg350_1
        del arg860_1
        del arg861_1
        del buf437
        buf440 = buf436; del buf436  # reuse
        # Source Nodes: [sp_294, sp_295], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg351_1, buf440, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg351_1
        # Source Nodes: [sp_294, sp_295], Original ATen: [aten.add, aten.convolution]
        buf441 = extern_kernels.convolution(buf439, buf440, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf441, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf439
        buf442 = reinterpret_tensor(buf444, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_296, sp_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf441, arg863_1, arg864_1, arg352_1, arg353_1, buf442, 163072, grid=grid(163072), stream=stream0)
        del arg352_1
        del arg353_1
        del arg863_1
        del arg864_1
        buf443 = reinterpret_tensor(buf444, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_43], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf430, buf443, 163072, grid=grid(163072), stream=stream0)
        buf445 = reinterpret_tensor(buf430, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf430  # reuse
        # Source Nodes: [out_180], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf444, buf445, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf434
        del buf438
        del buf442
        del buf443
        del buf444
        # Source Nodes: [out_180], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf445, arg354_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg354_1
        buf447 = buf428; del buf428  # reuse
        # Source Nodes: [out_181, out_182, shortcut_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf447, buf446, arg866_1, arg867_1, arg355_1, arg356_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg355_1
        del arg356_1
        del arg866_1
        del arg867_1
        del buf446
        # Source Nodes: [out_184], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, arg357_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg357_1
        buf449 = buf448; del buf448  # reuse
        # Source Nodes: [out_185, out_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf449, arg869_1, arg870_1, arg358_1, arg359_1, 652288, grid=grid(652288), stream=stream0)
        del arg358_1
        del arg359_1
        del arg869_1
        del arg870_1
        buf450 = reinterpret_tensor(buf441, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf441  # reuse
        # Source Nodes: [sp_300], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf449, buf450, 832, 196, grid=grid(832, 196), stream=stream0)
        buf451 = buf440; del buf440  # reuse
        # Source Nodes: [sp_300], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg360_1, buf451, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg360_1
        # Source Nodes: [sp_300], Original ATen: [aten.convolution]
        buf452 = extern_kernels.convolution(buf450, buf451, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf452, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf463 = reinterpret_tensor(buf445, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf445  # reuse
        buf453 = reinterpret_tensor(buf463, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf454 = buf450; del buf450  # reuse
        # Source Nodes: [sp_301, sp_302, sp_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf452, arg872_1, arg873_1, arg361_1, arg362_1, buf449, buf453, buf454, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg361_1
        del arg362_1
        del arg872_1
        del arg873_1
        del buf452
        buf455 = buf451; del buf451  # reuse
        # Source Nodes: [sp_303, sp_304], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg363_1, buf455, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg363_1
        # Source Nodes: [sp_303, sp_304], Original ATen: [aten.add, aten.convolution]
        buf456 = extern_kernels.convolution(buf454, buf455, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf457 = reinterpret_tensor(buf463, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf458 = buf454; del buf454  # reuse
        # Source Nodes: [sp_305, sp_306, sp_307], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf456, arg875_1, arg876_1, arg364_1, arg365_1, buf449, buf457, buf458, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg364_1
        del arg365_1
        del arg875_1
        del arg876_1
        del buf456
        buf459 = buf455; del buf455  # reuse
        # Source Nodes: [sp_307, sp_308], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg366_1, buf459, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg366_1
        # Source Nodes: [sp_307, sp_308], Original ATen: [aten.add, aten.convolution]
        buf460 = extern_kernels.convolution(buf458, buf459, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf460, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf458
        buf461 = reinterpret_tensor(buf463, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_309, sp_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf460, arg878_1, arg879_1, arg367_1, arg368_1, buf461, 163072, grid=grid(163072), stream=stream0)
        del arg367_1
        del arg368_1
        del arg878_1
        del arg879_1
        buf462 = reinterpret_tensor(buf463, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_42], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf449, buf462, 163072, grid=grid(163072), stream=stream0)
        buf464 = reinterpret_tensor(buf449, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf449  # reuse
        # Source Nodes: [out_188], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf463, buf464, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf453
        del buf457
        del buf461
        del buf462
        del buf463
        # Source Nodes: [out_188], Original ATen: [aten.convolution]
        buf465 = extern_kernels.convolution(buf464, arg369_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf465, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg369_1
        buf466 = buf447; del buf447  # reuse
        # Source Nodes: [out_189, out_190, shortcut_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf466, buf465, arg881_1, arg882_1, arg370_1, arg371_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg370_1
        del arg371_1
        del arg881_1
        del arg882_1
        del buf465
        # Source Nodes: [out_192], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, arg372_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg372_1
        buf468 = buf467; del buf467  # reuse
        # Source Nodes: [out_193, out_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf468, arg884_1, arg885_1, arg373_1, arg374_1, 652288, grid=grid(652288), stream=stream0)
        del arg373_1
        del arg374_1
        del arg884_1
        del arg885_1
        buf469 = reinterpret_tensor(buf460, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf460  # reuse
        # Source Nodes: [sp_313], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf468, buf469, 832, 196, grid=grid(832, 196), stream=stream0)
        buf470 = buf459; del buf459  # reuse
        # Source Nodes: [sp_313], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg375_1, buf470, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg375_1
        # Source Nodes: [sp_313], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf469, buf470, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf482 = reinterpret_tensor(buf464, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf464  # reuse
        buf472 = reinterpret_tensor(buf482, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf473 = buf469; del buf469  # reuse
        # Source Nodes: [sp_314, sp_315, sp_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf471, arg887_1, arg888_1, arg376_1, arg377_1, buf468, buf472, buf473, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg376_1
        del arg377_1
        del arg887_1
        del arg888_1
        del buf471
        buf474 = buf470; del buf470  # reuse
        # Source Nodes: [sp_316, sp_317], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg378_1, buf474, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg378_1
        # Source Nodes: [sp_316, sp_317], Original ATen: [aten.add, aten.convolution]
        buf475 = extern_kernels.convolution(buf473, buf474, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf476 = reinterpret_tensor(buf482, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf477 = buf473; del buf473  # reuse
        # Source Nodes: [sp_318, sp_319, sp_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf475, arg890_1, arg891_1, arg379_1, arg380_1, buf468, buf476, buf477, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg379_1
        del arg380_1
        del arg890_1
        del arg891_1
        del buf475
        buf478 = buf474; del buf474  # reuse
        # Source Nodes: [sp_320, sp_321], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg381_1, buf478, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg381_1
        # Source Nodes: [sp_320, sp_321], Original ATen: [aten.add, aten.convolution]
        buf479 = extern_kernels.convolution(buf477, buf478, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf477
        buf480 = reinterpret_tensor(buf482, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_322, sp_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf479, arg893_1, arg894_1, arg382_1, arg383_1, buf480, 163072, grid=grid(163072), stream=stream0)
        del arg382_1
        del arg383_1
        del arg893_1
        del arg894_1
        buf481 = reinterpret_tensor(buf482, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_41], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf468, buf481, 163072, grid=grid(163072), stream=stream0)
        buf483 = reinterpret_tensor(buf468, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf468  # reuse
        # Source Nodes: [out_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf482, buf483, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf472
        del buf476
        del buf480
        del buf481
        del buf482
        # Source Nodes: [out_196], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, arg384_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg384_1
        buf485 = buf466; del buf466  # reuse
        # Source Nodes: [out_197, out_198, shortcut_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf485, buf484, arg896_1, arg897_1, arg385_1, arg386_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg385_1
        del arg386_1
        del arg896_1
        del arg897_1
        del buf484
        # Source Nodes: [out_200], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf485, arg387_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg387_1
        buf487 = buf486; del buf486  # reuse
        # Source Nodes: [out_201, out_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf487, arg899_1, arg900_1, arg388_1, arg389_1, 652288, grid=grid(652288), stream=stream0)
        del arg388_1
        del arg389_1
        del arg899_1
        del arg900_1
        buf488 = reinterpret_tensor(buf479, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf479  # reuse
        # Source Nodes: [sp_326], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf487, buf488, 832, 196, grid=grid(832, 196), stream=stream0)
        buf489 = buf478; del buf478  # reuse
        # Source Nodes: [sp_326], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg390_1, buf489, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg390_1
        # Source Nodes: [sp_326], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf488, buf489, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf501 = reinterpret_tensor(buf483, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf483  # reuse
        buf491 = reinterpret_tensor(buf501, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf492 = buf488; del buf488  # reuse
        # Source Nodes: [sp_327, sp_328, sp_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf490, arg902_1, arg903_1, arg391_1, arg392_1, buf487, buf491, buf492, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg391_1
        del arg392_1
        del arg902_1
        del arg903_1
        del buf490
        buf493 = buf489; del buf489  # reuse
        # Source Nodes: [sp_329, sp_330], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg393_1, buf493, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg393_1
        # Source Nodes: [sp_329, sp_330], Original ATen: [aten.add, aten.convolution]
        buf494 = extern_kernels.convolution(buf492, buf493, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf494, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf495 = reinterpret_tensor(buf501, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf496 = buf492; del buf492  # reuse
        # Source Nodes: [sp_331, sp_332, sp_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf494, arg905_1, arg906_1, arg394_1, arg395_1, buf487, buf495, buf496, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg394_1
        del arg395_1
        del arg905_1
        del arg906_1
        del buf494
        buf497 = buf493; del buf493  # reuse
        # Source Nodes: [sp_333, sp_334], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg396_1, buf497, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg396_1
        # Source Nodes: [sp_333, sp_334], Original ATen: [aten.add, aten.convolution]
        buf498 = extern_kernels.convolution(buf496, buf497, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf496
        buf499 = reinterpret_tensor(buf501, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_335, sp_336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf498, arg908_1, arg909_1, arg397_1, arg398_1, buf499, 163072, grid=grid(163072), stream=stream0)
        del arg397_1
        del arg398_1
        del arg908_1
        del arg909_1
        buf500 = reinterpret_tensor(buf501, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_40], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf487, buf500, 163072, grid=grid(163072), stream=stream0)
        buf502 = reinterpret_tensor(buf487, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf487  # reuse
        # Source Nodes: [out_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf501, buf502, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf491
        del buf495
        del buf499
        del buf500
        del buf501
        # Source Nodes: [out_204], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf502, arg399_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf503, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg399_1
        buf504 = buf485; del buf485  # reuse
        # Source Nodes: [out_205, out_206, shortcut_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf504, buf503, arg911_1, arg912_1, arg400_1, arg401_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg400_1
        del arg401_1
        del arg911_1
        del arg912_1
        del buf503
        # Source Nodes: [out_208], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf504, arg402_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf505, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg402_1
        buf506 = buf505; del buf505  # reuse
        # Source Nodes: [out_209, out_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf506, arg914_1, arg915_1, arg403_1, arg404_1, 652288, grid=grid(652288), stream=stream0)
        del arg403_1
        del arg404_1
        del arg914_1
        del arg915_1
        buf507 = reinterpret_tensor(buf498, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf498  # reuse
        # Source Nodes: [sp_339], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf506, buf507, 832, 196, grid=grid(832, 196), stream=stream0)
        buf508 = buf497; del buf497  # reuse
        # Source Nodes: [sp_339], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg405_1, buf508, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg405_1
        # Source Nodes: [sp_339], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf507, buf508, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf520 = reinterpret_tensor(buf502, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf502  # reuse
        buf510 = reinterpret_tensor(buf520, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf511 = buf507; del buf507  # reuse
        # Source Nodes: [sp_340, sp_341, sp_342], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf509, arg917_1, arg918_1, arg406_1, arg407_1, buf506, buf510, buf511, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg406_1
        del arg407_1
        del arg917_1
        del arg918_1
        del buf509
        buf512 = buf508; del buf508  # reuse
        # Source Nodes: [sp_342, sp_343], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg408_1, buf512, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg408_1
        # Source Nodes: [sp_342, sp_343], Original ATen: [aten.add, aten.convolution]
        buf513 = extern_kernels.convolution(buf511, buf512, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf514 = reinterpret_tensor(buf520, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf515 = buf511; del buf511  # reuse
        # Source Nodes: [sp_344, sp_345, sp_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf513, arg920_1, arg921_1, arg409_1, arg410_1, buf506, buf514, buf515, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg409_1
        del arg410_1
        del arg920_1
        del arg921_1
        del buf513
        buf516 = buf512; del buf512  # reuse
        # Source Nodes: [sp_346, sp_347], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg411_1, buf516, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg411_1
        # Source Nodes: [sp_346, sp_347], Original ATen: [aten.add, aten.convolution]
        buf517 = extern_kernels.convolution(buf515, buf516, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf515
        buf518 = reinterpret_tensor(buf520, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_348, sp_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf517, arg923_1, arg924_1, arg412_1, arg413_1, buf518, 163072, grid=grid(163072), stream=stream0)
        del arg412_1
        del arg413_1
        del arg923_1
        del arg924_1
        buf519 = reinterpret_tensor(buf520, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_39], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf506, buf519, 163072, grid=grid(163072), stream=stream0)
        buf521 = reinterpret_tensor(buf506, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf506  # reuse
        # Source Nodes: [out_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf520, buf521, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf510
        del buf514
        del buf518
        del buf519
        del buf520
        # Source Nodes: [out_212], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, arg414_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg414_1
        buf523 = buf504; del buf504  # reuse
        # Source Nodes: [out_213, out_214, shortcut_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf523, buf522, arg926_1, arg927_1, arg415_1, arg416_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg415_1
        del arg416_1
        del arg926_1
        del arg927_1
        del buf522
        # Source Nodes: [out_216], Original ATen: [aten.convolution]
        buf524 = extern_kernels.convolution(buf523, arg417_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf524, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg417_1
        buf525 = buf524; del buf524  # reuse
        # Source Nodes: [out_217, out_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf525, arg929_1, arg930_1, arg418_1, arg419_1, 652288, grid=grid(652288), stream=stream0)
        del arg418_1
        del arg419_1
        del arg929_1
        del arg930_1
        buf526 = reinterpret_tensor(buf517, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf517  # reuse
        # Source Nodes: [sp_352], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf525, buf526, 832, 196, grid=grid(832, 196), stream=stream0)
        buf527 = buf516; del buf516  # reuse
        # Source Nodes: [sp_352], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg420_1, buf527, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg420_1
        # Source Nodes: [sp_352], Original ATen: [aten.convolution]
        buf528 = extern_kernels.convolution(buf526, buf527, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf528, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf539 = reinterpret_tensor(buf521, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf521  # reuse
        buf529 = reinterpret_tensor(buf539, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf530 = buf526; del buf526  # reuse
        # Source Nodes: [sp_353, sp_354, sp_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf528, arg932_1, arg933_1, arg421_1, arg422_1, buf525, buf529, buf530, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg421_1
        del arg422_1
        del arg932_1
        del arg933_1
        del buf528
        buf531 = buf527; del buf527  # reuse
        # Source Nodes: [sp_355, sp_356], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg423_1, buf531, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg423_1
        # Source Nodes: [sp_355, sp_356], Original ATen: [aten.add, aten.convolution]
        buf532 = extern_kernels.convolution(buf530, buf531, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf532, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf533 = reinterpret_tensor(buf539, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf534 = buf530; del buf530  # reuse
        # Source Nodes: [sp_357, sp_358, sp_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf532, arg935_1, arg936_1, arg424_1, arg425_1, buf525, buf533, buf534, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg424_1
        del arg425_1
        del arg935_1
        del arg936_1
        del buf532
        buf535 = buf531; del buf531  # reuse
        # Source Nodes: [sp_359, sp_360], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg426_1, buf535, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg426_1
        # Source Nodes: [sp_359, sp_360], Original ATen: [aten.add, aten.convolution]
        buf536 = extern_kernels.convolution(buf534, buf535, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf534
        buf537 = reinterpret_tensor(buf539, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_361, sp_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf536, arg938_1, arg939_1, arg427_1, arg428_1, buf537, 163072, grid=grid(163072), stream=stream0)
        del arg427_1
        del arg428_1
        del arg938_1
        del arg939_1
        buf538 = reinterpret_tensor(buf539, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_38], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf525, buf538, 163072, grid=grid(163072), stream=stream0)
        buf540 = reinterpret_tensor(buf525, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf525  # reuse
        # Source Nodes: [out_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf539, buf540, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf529
        del buf533
        del buf537
        del buf538
        del buf539
        # Source Nodes: [out_220], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf540, arg429_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg429_1
        buf542 = buf523; del buf523  # reuse
        # Source Nodes: [out_221, out_222, shortcut_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf542, buf541, arg941_1, arg942_1, arg430_1, arg431_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg430_1
        del arg431_1
        del arg941_1
        del arg942_1
        del buf541
        # Source Nodes: [out_224], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf542, arg432_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf543, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg432_1
        buf544 = buf543; del buf543  # reuse
        # Source Nodes: [out_225, out_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf544, arg944_1, arg945_1, arg433_1, arg434_1, 652288, grid=grid(652288), stream=stream0)
        del arg433_1
        del arg434_1
        del arg944_1
        del arg945_1
        buf545 = reinterpret_tensor(buf536, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf536  # reuse
        # Source Nodes: [sp_365], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf544, buf545, 832, 196, grid=grid(832, 196), stream=stream0)
        buf546 = buf535; del buf535  # reuse
        # Source Nodes: [sp_365], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg435_1, buf546, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg435_1
        # Source Nodes: [sp_365], Original ATen: [aten.convolution]
        buf547 = extern_kernels.convolution(buf545, buf546, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf558 = reinterpret_tensor(buf540, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf540  # reuse
        buf548 = reinterpret_tensor(buf558, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf549 = buf545; del buf545  # reuse
        # Source Nodes: [sp_366, sp_367, sp_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf547, arg947_1, arg948_1, arg436_1, arg437_1, buf544, buf548, buf549, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg436_1
        del arg437_1
        del arg947_1
        del arg948_1
        del buf547
        buf550 = buf546; del buf546  # reuse
        # Source Nodes: [sp_368, sp_369], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg438_1, buf550, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg438_1
        # Source Nodes: [sp_368, sp_369], Original ATen: [aten.add, aten.convolution]
        buf551 = extern_kernels.convolution(buf549, buf550, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf552 = reinterpret_tensor(buf558, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf553 = buf549; del buf549  # reuse
        # Source Nodes: [sp_370, sp_371, sp_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf551, arg950_1, arg951_1, arg439_1, arg440_1, buf544, buf552, buf553, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg439_1
        del arg440_1
        del arg950_1
        del arg951_1
        del buf551
        buf554 = buf550; del buf550  # reuse
        # Source Nodes: [sp_372, sp_373], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg441_1, buf554, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg441_1
        # Source Nodes: [sp_372, sp_373], Original ATen: [aten.add, aten.convolution]
        buf555 = extern_kernels.convolution(buf553, buf554, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf555, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf553
        buf556 = reinterpret_tensor(buf558, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_374, sp_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf555, arg953_1, arg954_1, arg442_1, arg443_1, buf556, 163072, grid=grid(163072), stream=stream0)
        del arg442_1
        del arg443_1
        del arg953_1
        del arg954_1
        buf557 = reinterpret_tensor(buf558, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_37], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf544, buf557, 163072, grid=grid(163072), stream=stream0)
        buf559 = reinterpret_tensor(buf544, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf544  # reuse
        # Source Nodes: [out_228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf558, buf559, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf548
        del buf552
        del buf556
        del buf557
        del buf558
        # Source Nodes: [out_228], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf559, arg444_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg444_1
        buf561 = buf542; del buf542  # reuse
        # Source Nodes: [out_229, out_230, shortcut_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf561, buf560, arg956_1, arg957_1, arg445_1, arg446_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg445_1
        del arg446_1
        del arg956_1
        del arg957_1
        del buf560
        # Source Nodes: [out_232], Original ATen: [aten.convolution]
        buf562 = extern_kernels.convolution(buf561, arg447_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (8, 416, 14, 14), (81536, 196, 14, 1))
        del arg447_1
        buf563 = buf562; del buf562  # reuse
        # Source Nodes: [out_233, out_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf563, arg959_1, arg960_1, arg448_1, arg449_1, 652288, grid=grid(652288), stream=stream0)
        del arg448_1
        del arg449_1
        del arg959_1
        del arg960_1
        buf564 = reinterpret_tensor(buf555, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf555  # reuse
        # Source Nodes: [sp_378], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf563, buf564, 832, 196, grid=grid(832, 196), stream=stream0)
        buf565 = buf554; del buf554  # reuse
        # Source Nodes: [sp_378], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(arg450_1, buf565, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg450_1
        # Source Nodes: [sp_378], Original ATen: [aten.convolution]
        buf566 = extern_kernels.convolution(buf564, buf565, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf577 = reinterpret_tensor(buf559, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf559  # reuse
        buf567 = reinterpret_tensor(buf577, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf568 = buf564; del buf564  # reuse
        # Source Nodes: [sp_379, sp_380, sp_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_43.run(buf566, arg962_1, arg963_1, arg451_1, arg452_1, buf563, buf567, buf568, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg451_1
        del arg452_1
        del arg962_1
        del arg963_1
        del buf566
        buf569 = buf565; del buf565  # reuse
        # Source Nodes: [sp_381, sp_382], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg453_1, buf569, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg453_1
        # Source Nodes: [sp_381, sp_382], Original ATen: [aten.add, aten.convolution]
        buf570 = extern_kernels.convolution(buf568, buf569, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf571 = reinterpret_tensor(buf577, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        buf572 = buf568; del buf568  # reuse
        # Source Nodes: [sp_383, sp_384, sp_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf570, arg965_1, arg966_1, arg454_1, arg455_1, buf563, buf571, buf572, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg454_1
        del arg455_1
        del arg965_1
        del arg966_1
        del buf570
        buf573 = buf569; del buf569  # reuse
        # Source Nodes: [sp_385, sp_386], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_34.run(arg456_1, buf573, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg456_1
        # Source Nodes: [sp_385, sp_386], Original ATen: [aten.add, aten.convolution]
        buf574 = extern_kernels.convolution(buf572, buf573, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (8, 104, 14, 14), (20384, 196, 14, 1))
        del buf572
        del buf573
        buf575 = reinterpret_tensor(buf577, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Source Nodes: [sp_387, sp_388], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf574, arg968_1, arg969_1, arg457_1, arg458_1, buf575, 163072, grid=grid(163072), stream=stream0)
        del arg457_1
        del arg458_1
        del arg968_1
        del arg969_1
        del buf574
        buf576 = reinterpret_tensor(buf577, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Source Nodes: [cat_36], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf563, buf576, 163072, grid=grid(163072), stream=stream0)
        buf578 = reinterpret_tensor(buf563, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf563  # reuse
        # Source Nodes: [out_236], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf577, buf578, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf567
        del buf571
        del buf575
        del buf576
        del buf577
        # Source Nodes: [out_236], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf578, arg459_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg459_1
        del buf578
        buf580 = buf561; del buf561  # reuse
        # Source Nodes: [out_237, out_238, shortcut_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf580, buf579, arg971_1, arg972_1, arg460_1, arg461_1, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del arg460_1
        del arg461_1
        del arg971_1
        del arg972_1
        del buf579
        # Source Nodes: [out_240], Original ATen: [aten.convolution]
        buf581 = extern_kernels.convolution(buf580, arg462_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf581, (8, 832, 14, 14), (163072, 196, 14, 1))
        del arg462_1
        buf582 = buf581; del buf581  # reuse
        # Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf582, arg974_1, arg975_1, arg463_1, arg464_1, 1304576, grid=grid(1304576), stream=stream0)
        del arg463_1
        del arg464_1
        del arg974_1
        del arg975_1
        buf583 = reinterpret_tensor(buf135, (8, 208, 14, 14), (40768, 1, 2912, 208), 0); del buf135  # reuse
        # Source Nodes: [sp_391], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48.run(buf582, buf583, 1664, 196, grid=grid(1664, 196), stream=stream0)
        buf584 = empty_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_391], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(arg465_1, buf584, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg465_1
        # Source Nodes: [sp_391], Original ATen: [aten.convolution]
        buf585 = extern_kernels.convolution(buf583, buf584, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf585, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf586 = buf583; del buf583  # reuse
        # Source Nodes: [sp_395], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf582, buf586, 1664, 196, grid=grid(1664, 196), stream=stream0)
        buf587 = buf584; del buf584  # reuse
        # Source Nodes: [sp_395], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(arg468_1, buf587, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg468_1
        # Source Nodes: [sp_395], Original ATen: [aten.convolution]
        buf588 = extern_kernels.convolution(buf586, buf587, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf588, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf589 = buf586; del buf586  # reuse
        # Source Nodes: [sp_399], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf582, buf589, 1664, 196, grid=grid(1664, 196), stream=stream0)
        buf590 = buf587; del buf587  # reuse
        # Source Nodes: [sp_399], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(arg471_1, buf590, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg471_1
        # Source Nodes: [sp_399], Original ATen: [aten.convolution]
        buf591 = extern_kernels.convolution(buf589, buf590, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf591, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf596 = reinterpret_tensor(buf589, (8, 832, 7, 7), (40768, 49, 7, 1), 0); del buf589  # reuse
        buf592 = reinterpret_tensor(buf596, (8, 208, 7, 7), (40768, 49, 7, 1), 30576)  # alias
        # Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_52.run(buf582, buf592, 81536, grid=grid(81536), stream=stream0)
        del buf582
        buf593 = reinterpret_tensor(buf596, (8, 208, 7, 7), (40768, 49, 7, 1), 0)  # alias
        # Source Nodes: [sp_392, sp_393], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_53.run(buf585, arg977_1, arg978_1, arg466_1, arg467_1, buf593, 81536, grid=grid(81536), stream=stream0)
        del arg466_1
        del arg467_1
        del arg977_1
        del arg978_1
        del buf585
        buf594 = reinterpret_tensor(buf596, (8, 208, 7, 7), (40768, 49, 7, 1), 10192)  # alias
        # Source Nodes: [sp_396, sp_397], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_53.run(buf588, arg980_1, arg981_1, arg469_1, arg470_1, buf594, 81536, grid=grid(81536), stream=stream0)
        del arg469_1
        del arg470_1
        del arg980_1
        del arg981_1
        del buf588
        buf595 = reinterpret_tensor(buf596, (8, 208, 7, 7), (40768, 49, 7, 1), 20384)  # alias
        # Source Nodes: [sp_400, sp_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_53.run(buf591, arg983_1, arg984_1, arg472_1, arg473_1, buf595, 81536, grid=grid(81536), stream=stream0)
        del arg472_1
        del arg473_1
        del arg983_1
        del arg984_1
        buf597 = reinterpret_tensor(buf133, (8, 832, 7, 7), (40768, 1, 5824, 832), 0); del buf133  # reuse
        # Source Nodes: [out_244], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf596, buf597, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf592
        del buf593
        del buf594
        del buf595
        del buf596
        # Source Nodes: [out_244], Original ATen: [aten.convolution]
        buf598 = extern_kernels.convolution(buf597, arg474_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf598, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg474_1
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf599 = extern_kernels.convolution(buf580, arg477_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf599, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg477_1
        del buf580
        buf600 = buf598; del buf598  # reuse
        buf601 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_245, out_246, shortcut_34, shortcut_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_55.run(buf600, arg986_1, arg987_1, arg475_1, arg476_1, buf599, arg989_1, arg990_1, arg478_1, arg479_1, buf601, 16384, 49, grid=grid(16384, 49), stream=stream0)
        del arg475_1
        del arg476_1
        del arg478_1
        del arg479_1
        del arg986_1
        del arg987_1
        del arg989_1
        del arg990_1
        del buf599
        del buf600
        # Source Nodes: [out_248, shortcut_35], Original ATen: [aten.convolution, aten.relu]
        buf602 = extern_kernels.convolution(buf601, arg480_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (8, 832, 7, 7), (40768, 49, 7, 1))
        del arg480_1
        buf603 = buf602; del buf602  # reuse
        # Source Nodes: [out_249, out_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_56.run(buf603, arg992_1, arg993_1, arg481_1, arg482_1, 326144, grid=grid(326144), stream=stream0)
        del arg481_1
        del arg482_1
        del arg992_1
        del arg993_1
        buf604 = reinterpret_tensor(buf591, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf591  # reuse
        # Source Nodes: [sp_404], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf603, buf604, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf605 = buf590; del buf590  # reuse
        # Source Nodes: [sp_404], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(arg483_1, buf605, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg483_1
        # Source Nodes: [sp_404], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf604, buf605, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf617 = reinterpret_tensor(buf597, (8, 832, 7, 7), (40768, 49, 7, 1), 0); del buf597  # reuse
        buf607 = reinterpret_tensor(buf617, (8, 208, 7, 7), (40768, 49, 7, 1), 0)  # alias
        buf608 = buf604; del buf604  # reuse
        # Source Nodes: [sp_405, sp_406, sp_407], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_58.run(buf606, arg995_1, arg996_1, arg484_1, arg485_1, buf603, buf607, buf608, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg484_1
        del arg485_1
        del arg995_1
        del arg996_1
        del buf606
        buf609 = buf605; del buf605  # reuse
        # Source Nodes: [sp_407, sp_408], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_49.run(arg486_1, buf609, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg486_1
        # Source Nodes: [sp_407, sp_408], Original ATen: [aten.add, aten.convolution]
        buf610 = extern_kernels.convolution(buf608, buf609, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf610, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf611 = reinterpret_tensor(buf617, (8, 208, 7, 7), (40768, 49, 7, 1), 10192)  # alias
        buf612 = buf608; del buf608  # reuse
        # Source Nodes: [sp_409, sp_410, sp_411], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_59.run(buf610, arg998_1, arg999_1, arg487_1, arg488_1, buf603, buf611, buf612, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg487_1
        del arg488_1
        del arg998_1
        del arg999_1
        del buf610
        buf613 = buf609; del buf609  # reuse
        # Source Nodes: [sp_411, sp_412], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_49.run(arg489_1, buf613, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg489_1
        # Source Nodes: [sp_411, sp_412], Original ATen: [aten.add, aten.convolution]
        buf614 = extern_kernels.convolution(buf612, buf613, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf614, (8, 208, 7, 7), (10192, 49, 7, 1))
        del buf612
        buf615 = reinterpret_tensor(buf617, (8, 208, 7, 7), (40768, 49, 7, 1), 20384)  # alias
        # Source Nodes: [sp_413, sp_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_53.run(buf614, arg1001_1, arg1002_1, arg490_1, arg491_1, buf615, 81536, grid=grid(81536), stream=stream0)
        del arg1001_1
        del arg1002_1
        del arg490_1
        del arg491_1
        buf616 = reinterpret_tensor(buf617, (8, 208, 7, 7), (40768, 49, 7, 1), 30576)  # alias
        # Source Nodes: [cat_34], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf603, buf616, 81536, grid=grid(81536), stream=stream0)
        buf618 = reinterpret_tensor(buf603, (8, 832, 7, 7), (40768, 1, 5824, 832), 0); del buf603  # reuse
        # Source Nodes: [out_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf617, buf618, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf607
        del buf611
        del buf615
        del buf616
        del buf617
        # Source Nodes: [out_252], Original ATen: [aten.convolution]
        buf619 = extern_kernels.convolution(buf618, arg492_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf619, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg492_1
        buf620 = buf601; del buf601  # reuse
        # Source Nodes: [out_253, out_254, shortcut_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_61.run(buf620, buf619, arg1004_1, arg1005_1, arg493_1, arg494_1, 392, 2048, grid=grid(392, 2048), stream=stream0)
        del arg1004_1
        del arg1005_1
        del arg493_1
        del arg494_1
        del buf619
        # Source Nodes: [out_256], Original ATen: [aten.convolution]
        buf621 = extern_kernels.convolution(buf620, arg495_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf621, (8, 832, 7, 7), (40768, 49, 7, 1))
        del arg495_1
        buf622 = buf621; del buf621  # reuse
        # Source Nodes: [out_257, out_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_56.run(buf622, arg1007_1, arg1008_1, arg496_1, arg497_1, 326144, grid=grid(326144), stream=stream0)
        del arg1007_1
        del arg1008_1
        del arg496_1
        del arg497_1
        buf623 = reinterpret_tensor(buf614, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf614  # reuse
        # Source Nodes: [sp_417], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf622, buf623, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf624 = buf613; del buf613  # reuse
        # Source Nodes: [sp_417], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(arg498_1, buf624, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg498_1
        # Source Nodes: [sp_417], Original ATen: [aten.convolution]
        buf625 = extern_kernels.convolution(buf623, buf624, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf625, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf636 = reinterpret_tensor(buf618, (8, 832, 7, 7), (40768, 49, 7, 1), 0); del buf618  # reuse
        buf626 = reinterpret_tensor(buf636, (8, 208, 7, 7), (40768, 49, 7, 1), 0)  # alias
        buf627 = buf623; del buf623  # reuse
        # Source Nodes: [sp_418, sp_419, sp_420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_58.run(buf625, arg1010_1, arg1011_1, arg499_1, arg500_1, buf622, buf626, buf627, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg1010_1
        del arg1011_1
        del arg499_1
        del arg500_1
        del buf625
        buf628 = buf624; del buf624  # reuse
        # Source Nodes: [sp_420, sp_421], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_49.run(arg501_1, buf628, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg501_1
        # Source Nodes: [sp_420, sp_421], Original ATen: [aten.add, aten.convolution]
        buf629 = extern_kernels.convolution(buf627, buf628, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf629, (8, 208, 7, 7), (10192, 49, 7, 1))
        buf630 = reinterpret_tensor(buf636, (8, 208, 7, 7), (40768, 49, 7, 1), 10192)  # alias
        buf631 = buf627; del buf627  # reuse
        # Source Nodes: [sp_422, sp_423, sp_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_59.run(buf629, arg1013_1, arg1014_1, arg502_1, arg503_1, buf622, buf630, buf631, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg1013_1
        del arg1014_1
        del arg502_1
        del arg503_1
        del buf629
        buf632 = buf628; del buf628  # reuse
        # Source Nodes: [sp_424, sp_425], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_49.run(arg504_1, buf632, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg504_1
        # Source Nodes: [sp_424, sp_425], Original ATen: [aten.add, aten.convolution]
        buf633 = extern_kernels.convolution(buf631, buf632, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (8, 208, 7, 7), (10192, 49, 7, 1))
        del buf631
        del buf632
        buf634 = reinterpret_tensor(buf636, (8, 208, 7, 7), (40768, 49, 7, 1), 20384)  # alias
        # Source Nodes: [sp_426, sp_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_53.run(buf633, arg1016_1, arg1017_1, arg505_1, arg506_1, buf634, 81536, grid=grid(81536), stream=stream0)
        del arg1016_1
        del arg1017_1
        del arg505_1
        del arg506_1
        del buf633
        buf635 = reinterpret_tensor(buf636, (8, 208, 7, 7), (40768, 49, 7, 1), 30576)  # alias
        # Source Nodes: [cat_33], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf622, buf635, 81536, grid=grid(81536), stream=stream0)
        buf637 = reinterpret_tensor(buf622, (8, 832, 7, 7), (40768, 1, 5824, 832), 0); del buf622  # reuse
        # Source Nodes: [out_260], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf636, buf637, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf626
        del buf630
        del buf634
        del buf635
        del buf636
        # Source Nodes: [out_260], Original ATen: [aten.convolution]
        buf638 = extern_kernels.convolution(buf637, arg507_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf638, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg507_1
        del buf637
        buf639 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf640 = reinterpret_tensor(buf639, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf639  # reuse
        # Source Nodes: [out_261, out_262, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_62.run(buf640, buf638, arg1019_1, arg1020_1, arg508_1, arg509_1, buf620, 16384, 49, grid=grid(16384), stream=stream0)
        del arg1019_1
        del arg1020_1
        del arg508_1
        del arg509_1
        del buf620
        del buf638
        buf641 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg511_1, reinterpret_tensor(buf640, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg510_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf641)
        del arg510_1
        del arg511_1
        return (buf641, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((104, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((104, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((104, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((208, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((416, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((832, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((832, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((832, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg515_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg518_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg521_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg524_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg527_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg530_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg533_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg536_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg539_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg542_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg545_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg548_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg551_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg554_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg557_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg560_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg563_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg566_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg569_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg572_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg575_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg578_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg581_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg584_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg587_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg590_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg593_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg596_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg599_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg602_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg605_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg608_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg611_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg614_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg617_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg620_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg623_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg626_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg629_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg632_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg635_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg638_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg641_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg644_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg647_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg650_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg653_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg656_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg659_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg662_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg665_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg668_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg671_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg674_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg677_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg680_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg683_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg686_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg689_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg692_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg695_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg698_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg701_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg704_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg707_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg710_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg713_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg716_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg719_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg722_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg725_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg728_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg731_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg734_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg737_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg740_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg743_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg746_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg749_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg752_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg755_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg758_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg761_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg764_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg767_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg770_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg773_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg776_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg779_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg782_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg785_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg788_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg791_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg794_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg797_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg800_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg803_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg806_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg809_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg812_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg815_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg818_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg821_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg824_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg827_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg830_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg833_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg836_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg839_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg842_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg845_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg848_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg851_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg853_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg854_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg855_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg856_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg857_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg858_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg859_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg860_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg861_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg862_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg863_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg864_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg865_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg866_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg867_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg868_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg869_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg870_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg871_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg872_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg873_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg874_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg875_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg876_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg877_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg878_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg879_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg880_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg881_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg882_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg883_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg884_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg885_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg886_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg887_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg888_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg889_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg890_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg891_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg892_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg893_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg894_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg895_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg896_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg897_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg898_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg899_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg900_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg901_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg902_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg903_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg904_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg905_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg906_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg907_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg908_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg909_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg910_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg911_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg912_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg913_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg914_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg915_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg916_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg917_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg918_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg919_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg920_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg921_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg922_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg923_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg924_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg925_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg926_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg927_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg928_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg929_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg930_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg931_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg932_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg933_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg934_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg935_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg936_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg937_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg938_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg939_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg940_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg941_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg942_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg943_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg944_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg945_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg946_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg947_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg948_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg949_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg950_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg951_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg952_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg953_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg954_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg955_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg956_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg957_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg958_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg959_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg960_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg961_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg962_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg963_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg964_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg965_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg966_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg967_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg968_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg969_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg970_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg971_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg972_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg973_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg974_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg975_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg976_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg977_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg978_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg979_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg980_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg981_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg982_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg983_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg984_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg985_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg986_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg987_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg988_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg989_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg990_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg991_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg992_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg993_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg994_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg995_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg996_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg997_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg998_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg999_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1000_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1001_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1002_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1003_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1004_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1005_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1006_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1007_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1008_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1009_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1010_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1011_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1012_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1013_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1014_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1015_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1016_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1017_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1018_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1019_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1020_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1021_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1022_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2net101_26w_4s', benchmark_compiled_module)
