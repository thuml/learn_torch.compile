
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
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
            tmp14.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
            tmp14.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
            tmp14.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
            tmp14.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
            tmp14.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(791800L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp14.store(out_ptr0 + static_cast<long>(x0));
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(791800L); x0<static_cast<long>(791804L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
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
                out_ptr0[static_cast<long>(x0)] = tmp10;
            }
        }
    }
}
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
    buf0 = empty((4, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_4, primals_13, reinterpret_tensor(primals_1, (197951, 512), (1, 197951), 0), alpha=1, beta=1, out=buf0)
    del primals_1
    del primals_4
    buf1 = empty((4, 512), device='cpu', dtype=torch.float32)
    cpp_fused_elu_0(c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    buf2 = empty((4, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [linear_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_5, buf1, reinterpret_tensor(primals_2, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf2)
    del primals_5
    buf3 = empty((4, 512), device='cpu', dtype=torch.float32)
    cpp_fused_elu_1(c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()))
    buf4 = empty((4, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [linear_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_6, buf3, reinterpret_tensor(primals_3, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf4)
    del primals_6
    buf5 = empty((4, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_elu_2(c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    buf6 = empty((4, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [linear_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_10, buf5, reinterpret_tensor(primals_7, (1024, 512), (1, 1024), 0), alpha=1, beta=1, out=buf6)
    del primals_10
    buf7 = empty((4, 512), device='cpu', dtype=torch.float32)
    cpp_fused_elu_3(c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    buf8 = empty((4, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [linear_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_11, buf7, reinterpret_tensor(primals_8, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf8)
    del primals_11
    buf9 = empty((4, 512), device='cpu', dtype=torch.float32)
    cpp_fused_elu_4(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    buf10 = empty((4, 197951), device='cpu', dtype=torch.float32)
    # Source Nodes: [linear_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_12, buf9, reinterpret_tensor(primals_9, (512, 197951), (1, 512), 0), alpha=1, beta=1, out=buf10)
    del primals_12
    buf11 = empty((4, 197951), device='cpu', dtype=torch.float32)
    cpp_fused_elu_5(c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    return (buf11, primals_13, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, reinterpret_tensor(primals_9, (197951, 512), (512, 1), 0), reinterpret_tensor(primals_8, (512, 512), (512, 1), 0), reinterpret_tensor(primals_7, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_3, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_2, (512, 512), (512, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, 197951), (197951, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((197951, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((197951, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((4, 197951), (197951, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nvidia_deeprecommender', benchmark_compiled_module)
