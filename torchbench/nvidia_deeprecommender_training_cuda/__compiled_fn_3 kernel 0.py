
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


# kernel path: /tmp/torchinductor_youkaichao/fk/cfkzskqbadxc6f2jgkxzlix7e4ygzf6p53fhxgo4hnsapuc5dn7i.py
# Source Nodes: [x], Original ATen: [aten.elu]
# x => expm1, gt, mul, mul_1, mul_2, where
triton_poi_fused_elu_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0507009873554805
    tmp4 = tmp0 * tmp3
    tmp5 = 1.0
    tmp6 = tmp0 * tmp5
    tmp7 = tl.math.expm1(tmp6)
    tmp8 = 1.7580993408473766
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp2, tmp4, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/wa/cwat3ucbazvhglt2oamxrtzh3f5b7wx763ihxyi3s4rjaygxoddj.py
# Source Nodes: [x_2], Original ATen: [aten.elu]
# x_2 => expm1_2, gt_2, mul_6, mul_7, mul_8, where_2
triton_poi_fused_elu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0507009873554805
    tmp4 = tmp0 * tmp3
    tmp5 = 1.0
    tmp6 = tmp0 * tmp5
    tmp7 = tl.math.expm1(tmp6)
    tmp8 = 1.7580993408473766
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp2, tmp4, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cuf52plmf4ecwbjhqyim7tdtqyodc3lpbd3st5ggjnaykoyfpmac.py
# Source Nodes: [pred], Original ATen: [aten.elu]
# pred => expm1_5, gt_5, mul_15, mul_16, mul_17, where_5
triton_poi_fused_elu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 791804
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0507009873554805
    tmp4 = tmp0 * tmp3
    tmp5 = 1.0
    tmp6 = tmp0 * tmp5
    tmp7 = tl.math.expm1(tmp6)
    tmp8 = 1.7580993408473766
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp2, tmp4, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13 = args
    args.clear()
    assert_size_stride(primals_1, (512, 197951), (197951, 1))
    assert_size_stride(primals_2, (512, 512), (512, 1))
    assert_size_stride(primals_3, (1024, 512), (512, 1))
    assert_size_stride(primals_4, (512, ), (1, ))
    assert_size_stride(primals_5, (512, ), (1, ))
    assert_size_stride(primals_6, (1024, ), (1, ))
    assert_size_stride(primals_7, (512, 1024), (1024, 1))
    assert_size_stride(primals_8, (512, 512), (512, 1))
    assert_size_stride(primals_9, (197951, 512), (512, 1))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (512, ), (1, ))
    assert_size_stride(primals_12, (197951, ), (1, ))
    assert_size_stride(primals_13, (4, 197951), (197951, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_4, primals_13, reinterpret_tensor(primals_1, (197951, 512), (1, 197951), 0), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_4
        buf1 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.elu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_elu_0.run(buf0, buf1, 2048, grid=grid(2048), stream=stream0)
        buf2 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf1, reinterpret_tensor(primals_2, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf2)
        del primals_5
        buf3 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten.elu]
        triton_poi_fused_elu_0.run(buf2, buf3, 2048, grid=grid(2048), stream=stream0)
        buf4 = empty((4, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, buf3, reinterpret_tensor(primals_3, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf4)
        del primals_6
        buf5 = empty((4, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_2], Original ATen: [aten.elu]
        triton_poi_fused_elu_1.run(buf4, buf5, 4096, grid=grid(4096), stream=stream0)
        buf6 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, buf5, reinterpret_tensor(primals_7, (1024, 512), (1, 1024), 0), alpha=1, beta=1, out=buf6)
        del primals_10
        buf7 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [z], Original ATen: [aten.elu]
        triton_poi_fused_elu_0.run(buf6, buf7, 2048, grid=grid(2048), stream=stream0)
        buf8 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf7, reinterpret_tensor(primals_8, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf8)
        del primals_11
        buf9 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [z_1], Original ATen: [aten.elu]
        triton_poi_fused_elu_0.run(buf8, buf9, 2048, grid=grid(2048), stream=stream0)
        buf10 = empty((4, 197951), device='cuda', dtype=torch.float32)
        # Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, buf9, reinterpret_tensor(primals_9, (512, 197951), (1, 512), 0), alpha=1, beta=1, out=buf10)
        del primals_12
        buf11 = empty((4, 197951), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.elu]
        triton_poi_fused_elu_2.run(buf10, buf11, 791804, grid=grid(791804), stream=stream0)
        return (buf11, primals_13, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, reinterpret_tensor(primals_9, (197951, 512), (512, 1), 0), reinterpret_tensor(primals_8, (512, 512), (512, 1), 0), reinterpret_tensor(primals_7, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_3, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_2, (512, 512), (512, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, 197951), (197951, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((197951, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((197951, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, 197951), (197951, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nvidia_deeprecommender', benchmark_compiled_module)
