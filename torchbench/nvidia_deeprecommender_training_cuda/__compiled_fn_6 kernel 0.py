
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


# kernel path: /tmp/torchinductor_youkaichao/3x/c3xii5rqb55tbwce6e6jotvc2plogucycrvsey73xmoqj5otkyp5.py
# Source Nodes: [pred], Original ATen: [aten.elu, aten.elu_backward]
# pred => mul_16
triton_poi_fused_elu_elu_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_elu_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 791804
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = 1.7580993408473766
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 * tmp4
    tmp9 = tl.exp(tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = 1.0507009873554805
    tmp12 = tmp3 * tmp11
    tmp13 = tl.where(tmp2, tmp10, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/zh/czhiocvlpujjwuw6ug3tekoecbezvhzz6jtbpxf6qpuniez3sccy.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 197951
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (197951 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (395902 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (593853 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6iy5zcwgxw3w7nnmfc72bwhbpsddovf3wmbnk2grxgwflcwgsl.py
# Source Nodes: [z_1], Original ATen: [aten.elu, aten.elu_backward]
# z_1 => mul_13
triton_poi_fused_elu_elu_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_elu_backward_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = 1.7580993408473766
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 * tmp4
    tmp9 = tl.exp(tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = 1.0507009873554805
    tmp12 = tmp3 * tmp11
    tmp13 = tl.where(tmp2, tmp10, tmp12)
    tl.store(in_out_ptr0 + (x0), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7euud44s364cqrtzgx773zurombpjslinsbizliq273egkctgte.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_3', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr0 + (512 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (1536 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/35/c35bafa4seany3s6vvpkc2bimprfzbbbkvzom4skqneestm7kzyn.py
# Source Nodes: [x_2], Original ATen: [aten.elu, aten.elu_backward]
# x_2 => mul_7
triton_poi_fused_elu_elu_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_elu_backward_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = 1.7580993408473766
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 * tmp4
    tmp9 = tl.exp(tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = 1.0507009873554805
    tmp12 = tmp3 * tmp11
    tmp13 = tl.where(tmp2, tmp10, tmp12)
    tl.store(in_out_ptr0 + (x0), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rt/crtcvptxbmggyrpszkkm4xo4fflgcgoioohyezgjj55ymsibgzpl.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (2048 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (3072 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_13, addmm, where, addmm_1, where_1, addmm_2, clone, addmm_3, where_3, addmm_4, where_4, addmm_5, permute_6, permute_10, permute_14, permute_18, permute_22, tangents_1 = args
    args.clear()
    assert_size_stride(primals_13, (4, 197951), (197951, 1))
    assert_size_stride(addmm, (4, 512), (512, 1))
    assert_size_stride(where, (4, 512), (512, 1))
    assert_size_stride(addmm_1, (4, 512), (512, 1))
    assert_size_stride(where_1, (4, 512), (512, 1))
    assert_size_stride(addmm_2, (4, 1024), (1024, 1))
    assert_size_stride(clone, (4, 1024), (1024, 1))
    assert_size_stride(addmm_3, (4, 512), (512, 1))
    assert_size_stride(where_3, (4, 512), (512, 1))
    assert_size_stride(addmm_4, (4, 512), (512, 1))
    assert_size_stride(where_4, (4, 512), (512, 1))
    assert_size_stride(addmm_5, (4, 197951), (197951, 1))
    assert_size_stride(permute_6, (197951, 512), (512, 1))
    assert_size_stride(permute_10, (512, 512), (512, 1))
    assert_size_stride(permute_14, (512, 1024), (1024, 1))
    assert_size_stride(permute_18, (1024, 512), (512, 1))
    assert_size_stride(permute_22, (512, 512), (512, 1))
    assert_size_stride(tangents_1, (4, 197951), (197951, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 197951), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.elu, aten.elu_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_elu_elu_backward_0.run(addmm_5, tangents_1, buf0, 791804, grid=grid(791804), stream=stream0)
        del addmm_5
        del tangents_1
        buf1 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, permute_6, out=buf1)
        del permute_6
        buf2 = empty((197951, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (197951, 4), (1, 197951), 0), where_4, out=buf2)
        del where_4
        buf3 = empty((197951, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_1.run(buf0, buf3, 197951, grid=grid(197951), stream=stream0)
        del buf0
        buf4 = buf1; del buf1  # reuse
        # Source Nodes: [z_1], Original ATen: [aten.elu, aten.elu_backward]
        triton_poi_fused_elu_elu_backward_2.run(buf4, addmm_4, 2048, grid=grid(2048), stream=stream0)
        del addmm_4
        buf5 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, permute_10, out=buf5)
        del permute_10
        buf6 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 4), (1, 512), 0), where_3, out=buf6)
        del where_3
        buf7 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_3.run(buf4, buf7, 512, grid=grid(512), stream=stream0)
        buf8 = buf5; del buf5  # reuse
        # Source Nodes: [z], Original ATen: [aten.elu, aten.elu_backward]
        triton_poi_fused_elu_elu_backward_2.run(buf8, addmm_3, 2048, grid=grid(2048), stream=stream0)
        del addmm_3
        buf9 = empty((4, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf8, permute_14, out=buf9)
        del permute_14
        buf10 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (512, 4), (1, 512), 0), clone, out=buf10)
        del clone
        buf11 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_3.run(buf8, buf11, 512, grid=grid(512), stream=stream0)
        buf12 = buf9; del buf9  # reuse
        # Source Nodes: [x_2], Original ATen: [aten.elu, aten.elu_backward]
        triton_poi_fused_elu_elu_backward_4.run(buf12, addmm_2, 4096, grid=grid(4096), stream=stream0)
        del addmm_2
        buf13 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf12, permute_18, out=buf13)
        del permute_18
        buf14 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (1024, 4), (1, 1024), 0), where_1, out=buf14)
        del where_1
        buf15 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_5.run(buf12, buf15, 1024, grid=grid(1024), stream=stream0)
        del buf12
        buf16 = buf13; del buf13  # reuse
        # Source Nodes: [x_1], Original ATen: [aten.elu, aten.elu_backward]
        triton_poi_fused_elu_elu_backward_2.run(buf16, addmm_1, 2048, grid=grid(2048), stream=stream0)
        del addmm_1
        buf17 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf16, permute_22, out=buf17)
        del permute_22
        buf18 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (512, 4), (1, 512), 0), where, out=buf18)
        del where
        buf19 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_3.run(buf16, buf19, 512, grid=grid(512), stream=stream0)
        del buf16
        buf20 = buf17; del buf17  # reuse
        # Source Nodes: [x], Original ATen: [aten.elu, aten.elu_backward]
        triton_poi_fused_elu_elu_backward_2.run(buf20, addmm, 2048, grid=grid(2048), stream=stream0)
        del addmm
        buf21 = empty((512, 197951), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (512, 4), (1, 512), 0), primals_13, out=buf21)
        del primals_13
        buf22 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_3.run(buf20, buf22, 512, grid=grid(512), stream=stream0)
        return (reinterpret_tensor(buf21, (512, 197951), (197951, 1), 0), reinterpret_tensor(buf18, (512, 512), (512, 1), 0), reinterpret_tensor(buf14, (1024, 512), (512, 1), 0), buf22, buf19, buf15, reinterpret_tensor(buf10, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf6, (512, 512), (512, 1), 0), reinterpret_tensor(buf2, (197951, 512), (512, 1), 0), buf11, buf7, buf3, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_13 = rand_strided((4, 197951), (197951, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    where = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    where_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_3 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    where_3 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    where_4 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((4, 197951), (197951, 1), device='cuda:0', dtype=torch.float32)
    permute_6 = rand_strided((197951, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_10 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_14 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_18 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_22 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 197951), (197951, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_13, addmm, where, addmm_1, where_1, addmm_2, clone, addmm_3, where_3, addmm_4, where_4, addmm_5, permute_6, permute_10, permute_14, permute_18, permute_22, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nvidia_deeprecommender', benchmark_compiled_module)
