
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


cpp_fused_elu_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 > tmp2);
            auto tmp4 = static_cast<float>(1.0507009873554805);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp0 * tmp5;
            auto tmp7 = static_cast<float>(1.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp0 * tmp8;
            auto tmp10 = tmp9.exp() - decltype(tmp9)(1);
            auto tmp11 = static_cast<float>(1.7580993408473766);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = decltype(tmp6)::blendv(tmp13, tmp6, tmp3);
            tmp14.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 > tmp2);
            auto tmp4 = static_cast<float>(1.0507009873554805);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp0 * tmp5;
            auto tmp7 = static_cast<float>(1.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp0 * tmp8;
            auto tmp10 = tmp9.exp() - decltype(tmp9)(1);
            auto tmp11 = static_cast<float>(1.7580993408473766);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = decltype(tmp6)::blendv(tmp13, tmp6, tmp3);
            tmp14.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 > tmp2);
            auto tmp4 = static_cast<float>(1.0507009873554805);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp0 * tmp5;
            auto tmp7 = static_cast<float>(1.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp0 * tmp8;
            auto tmp10 = tmp9.exp() - decltype(tmp9)(1);
            auto tmp11 = static_cast<float>(1.7580993408473766);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = decltype(tmp6)::blendv(tmp13, tmp6, tmp3);
            tmp14.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 > tmp2);
            auto tmp4 = static_cast<float>(1.0507009873554805);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp0 * tmp5;
            auto tmp7 = static_cast<float>(1.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp0 * tmp8;
            auto tmp10 = tmp9.exp() - decltype(tmp9)(1);
            auto tmp11 = static_cast<float>(1.7580993408473766);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = decltype(tmp6)::blendv(tmp13, tmp6, tmp3);
            tmp14.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 > tmp2);
            auto tmp4 = static_cast<float>(1.0507009873554805);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp0 * tmp5;
            auto tmp7 = static_cast<float>(1.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp0 * tmp8;
            auto tmp10 = tmp9.exp() - decltype(tmp9)(1);
            auto tmp11 = static_cast<float>(1.7580993408473766);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = decltype(tmp6)::blendv(tmp13, tmp6, tmp3);
            tmp14.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(791800L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 > tmp2);
                auto tmp4 = static_cast<float>(1.0507009873554805);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = static_cast<float>(1.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp0 * tmp8;
                auto tmp10 = tmp9.exp() - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(1.7580993408473766);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp10 * tmp12;
                auto tmp14 = decltype(tmp6)::blendv(tmp13, tmp6, tmp3);
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(791800L); x0<static_cast<long>(791804L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 > tmp1;
                auto tmp3 = static_cast<float>(1.0507009873554805);
                auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                auto tmp5 = static_cast<float>(1.0);
                auto tmp6 = decltype(tmp0)(tmp0 * tmp5);
                auto tmp7 = std::expm1(tmp6);
                auto tmp8 = static_cast<float>(1.7580993408473766);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = tmp2 ? tmp4 : tmp9;
                in_out_ptr0[static_cast<long>(x0)] = tmp10;
            }
        }
    }
}
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
    buf0 = empty((4, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg3_1, arg12_1, reinterpret_tensor(arg0_1, (197951, 512), (1, 197951), 0), alpha=1, beta=1, out=buf0)
    del arg0_1
    del arg12_1
    del arg3_1
    buf1 = buf0; del buf0  # reuse
    cpp_fused_elu_0(c_void_p(buf1.data_ptr()))
    buf2 = empty((4, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [linear_1, x], Original ATen: [aten.addmm, aten.elu]
    extern_kernels.addmm(arg4_1, buf1, reinterpret_tensor(arg1_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf2)
    del arg1_1
    del arg4_1
    buf3 = buf2; del buf2  # reuse
    cpp_fused_elu_1(c_void_p(buf3.data_ptr()))
    buf4 = empty((4, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [linear_2, x_1], Original ATen: [aten.addmm, aten.elu]
    extern_kernels.addmm(arg5_1, buf3, reinterpret_tensor(arg2_1, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf4)
    del arg2_1
    del arg5_1
    buf5 = buf4; del buf4  # reuse
    cpp_fused_elu_2(c_void_p(buf5.data_ptr()))
    buf6 = buf3; del buf3  # reuse
    # Source Nodes: [linear_3, x_2], Original ATen: [aten.addmm, aten.elu]
    extern_kernels.addmm(arg9_1, buf5, reinterpret_tensor(arg6_1, (1024, 512), (1, 1024), 0), alpha=1, beta=1, out=buf6)
    del arg6_1
    del arg9_1
    del buf5
    buf7 = buf6; del buf6  # reuse
    cpp_fused_elu_3(c_void_p(buf7.data_ptr()))
    buf8 = buf1; del buf1  # reuse
    # Source Nodes: [linear_4, z], Original ATen: [aten.addmm, aten.elu]
    extern_kernels.addmm(arg10_1, buf7, reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf8)
    del arg10_1
    del arg7_1
    del buf7
    buf9 = buf8; del buf8  # reuse
    cpp_fused_elu_4(c_void_p(buf9.data_ptr()))
    buf10 = empty((4, 197951), device='cpu', dtype=torch.float32)
    # Source Nodes: [linear_5, z_1], Original ATen: [aten.addmm, aten.elu]
    extern_kernels.addmm(arg11_1, buf9, reinterpret_tensor(arg8_1, (512, 197951), (1, 512), 0), alpha=1, beta=1, out=buf10)
    del arg11_1
    del arg8_1
    del buf9
    buf11 = buf10; del buf10  # reuse
    cpp_fused_elu_5(c_void_p(buf11.data_ptr()))
    return (buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, 197951), (197951, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((197951, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((197951, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((4, 197951), (197951, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nvidia_deeprecommender', benchmark_compiled_module)
