
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


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbjrriz4hdynlktahd25ssq3qrey22vh2rnldqerpph6xl6sg2t.py
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 1.0507009873554805
    tmp6 = tmp2 * tmp5
    tmp7 = 1.0
    tmp8 = tmp2 * tmp7
    tmp9 = tl.math.expm1(tmp8)
    tmp10 = 1.7580993408473766
    tmp11 = tmp9 * tmp10
    tmp12 = tl.where(tmp4, tmp6, tmp11)
    tl.store(in_out_ptr0 + (x2), tmp12, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/45/c45yyuo6th62j2zfewjy6txnarkklcsf6godjtgckeg5vczhtgip.py
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 1.0507009873554805
    tmp6 = tmp2 * tmp5
    tmp7 = 1.0
    tmp8 = tmp2 * tmp7
    tmp9 = tl.math.expm1(tmp8)
    tmp10 = 1.7580993408473766
    tmp11 = tmp9 * tmp10
    tmp12 = tl.where(tmp4, tmp6, tmp11)
    tl.store(in_out_ptr0 + (x2), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uy/cuyq43rvgfwgn3xtew2wrdqmyenuwhvwip4s3ksoddwgmyei75fe.py
# Source Nodes: [z_2], Original ATen: [aten.elu]
# z_2 => expm1_5, gt_5, mul_15, mul_16, mul_17, where_5
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 791804
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 197951
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 1.0507009873554805
    tmp6 = tmp2 * tmp5
    tmp7 = 1.0
    tmp8 = tmp2 * tmp7
    tmp9 = tl.math.expm1(tmp8)
    tmp10 = 1.7580993408473766
    tmp11 = tmp9 * tmp10
    tmp12 = tl.where(tmp4, tmp6, tmp11)
    tl.store(in_out_ptr0 + (x2), tmp12, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, 197951), (197951, 1))
    assert_size_stride(arg1_1, (512, 512), (512, 1))
    assert_size_stride(arg2_1, (1024, 512), (512, 1))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (1024, ), (1, ))
    assert_size_stride(arg6_1, (512, 1024), (1024, 1))
    assert_size_stride(arg7_1, (512, 512), (512, 1))
    assert_size_stride(arg8_1, (197951, 512), (512, 1))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (197951, ), (1, ))
    assert_size_stride(arg12_1, (4, 197951), (197951, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(arg12_1, reinterpret_tensor(arg0_1, (197951, 512), (1, 197951), 0), out=buf0)
        del arg0_1
        del arg12_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [x], Original ATen: [aten.elu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_elu_0.run(buf1, arg3_1, 2048, grid=grid(2048), stream=stream0)
        del arg3_1
        buf2 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.elu]
        extern_kernels.mm(buf1, reinterpret_tensor(arg1_1, (512, 512), (1, 512), 0), out=buf2)
        del arg1_1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [x_1], Original ATen: [aten.elu]
        triton_poi_fused_elu_0.run(buf3, arg4_1, 2048, grid=grid(2048), stream=stream0)
        del arg4_1
        buf4 = empty((4, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten.elu]
        extern_kernels.mm(buf3, reinterpret_tensor(arg2_1, (512, 1024), (1, 512), 0), out=buf4)
        del arg2_1
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [x_2], Original ATen: [aten.elu]
        triton_poi_fused_elu_1.run(buf5, arg5_1, 4096, grid=grid(4096), stream=stream0)
        del arg5_1
        buf6 = buf3; del buf3  # reuse
        # Source Nodes: [x_2], Original ATen: [aten.elu]
        extern_kernels.mm(buf5, reinterpret_tensor(arg6_1, (1024, 512), (1, 1024), 0), out=buf6)
        del arg6_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [z], Original ATen: [aten.elu]
        triton_poi_fused_elu_0.run(buf7, arg9_1, 2048, grid=grid(2048), stream=stream0)
        del arg9_1
        buf8 = buf1; del buf1  # reuse
        # Source Nodes: [z], Original ATen: [aten.elu]
        extern_kernels.mm(buf7, reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), out=buf8)
        del arg7_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Source Nodes: [z_1], Original ATen: [aten.elu]
        triton_poi_fused_elu_0.run(buf9, arg10_1, 2048, grid=grid(2048), stream=stream0)
        del arg10_1
        buf10 = empty((4, 197951), device='cuda', dtype=torch.float32)
        # Source Nodes: [z_1], Original ATen: [aten.elu]
        extern_kernels.mm(buf9, reinterpret_tensor(arg8_1, (512, 197951), (1, 512), 0), out=buf10)
        del arg8_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [z_2], Original ATen: [aten.elu]
        triton_poi_fused_elu_2.run(buf11, arg11_1, 791804, grid=grid(791804), stream=stream0)
        del arg11_1
        return (buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, 197951), (197951, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((197951, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((197951, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((4, 197951), (197951, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nvidia_deeprecommender', benchmark_compiled_module)
