
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


cpp_fused_elu_elu_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(791800L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = static_cast<float>(1.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = static_cast<float>(1.7580993408473766);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 * tmp9;
                auto tmp11 = tmp0 * tmp6;
                auto tmp12 = tmp11.exp();
                auto tmp13 = tmp10 * tmp12;
                auto tmp14 = static_cast<float>(1.0507009873554805);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp4 * tmp15;
                auto tmp17 = decltype(tmp13)::blendv(tmp16, tmp13, tmp3);
                tmp17.store(out_ptr0 + static_cast<long>(x0));
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(791800L); x0<static_cast<long>(791804L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp3 = in_ptr1[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = static_cast<float>(1.0);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = static_cast<float>(1.7580993408473766);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp8 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp9 = std::exp(tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = static_cast<float>(1.0507009873554805);
                auto tmp12 = decltype(tmp3)(tmp3 * tmp11);
                auto tmp13 = tmp2 ? tmp10 : tmp12;
                out_ptr0[static_cast<long>(x0)] = tmp13;
            }
        }
    }
}
''')


cpp_fused_elu_elu_backward_sum_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(197944L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(197951L + x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(395902L + x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(593853L + x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(out_ptr0 + static_cast<long>(x0));
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(197944L); x0<static_cast<long>(197951L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr0[static_cast<long>(197951L + x0)];
                auto tmp3 = in_ptr0[static_cast<long>(395902L + x0)];
                auto tmp5 = in_ptr0[static_cast<long>(593853L + x0)];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                out_ptr0[static_cast<long>(x0)] = tmp6;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = static_cast<float>(1.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = static_cast<float>(1.7580993408473766);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 * tmp6;
                    auto tmp12 = tmp11.exp();
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = static_cast<float>(1.0507009873554805);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp4 * tmp15;
                    auto tmp17 = decltype(tmp13)::blendv(tmp16, tmp13, tmp3);
                    tmp17.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_elu_elu_backward_sum_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1536L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = static_cast<float>(1.0);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = static_cast<float>(1.7580993408473766);
            auto tmp9 = at::vec::Vectorized<float>(tmp8);
            auto tmp10 = tmp7 * tmp9;
            auto tmp11 = tmp0 * tmp6;
            auto tmp12 = tmp11.exp();
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = static_cast<float>(1.0507009873554805);
            auto tmp15 = at::vec::Vectorized<float>(tmp14);
            auto tmp16 = tmp4 * tmp15;
            auto tmp17 = decltype(tmp13)::blendv(tmp16, tmp13, tmp3);
            tmp17.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_elu_backward_sum_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1536L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = static_cast<float>(1.0);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = static_cast<float>(1.7580993408473766);
            auto tmp9 = at::vec::Vectorized<float>(tmp8);
            auto tmp10 = tmp7 * tmp9;
            auto tmp11 = tmp0 * tmp6;
            auto tmp12 = tmp11.exp();
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = static_cast<float>(1.0507009873554805);
            auto tmp15 = at::vec::Vectorized<float>(tmp14);
            auto tmp16 = tmp4 * tmp15;
            auto tmp17 = decltype(tmp13)::blendv(tmp16, tmp13, tmp3);
            tmp17.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_elu_backward_sum_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(2048L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3072L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = static_cast<float>(1.0);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = static_cast<float>(1.7580993408473766);
            auto tmp9 = at::vec::Vectorized<float>(tmp8);
            auto tmp10 = tmp7 * tmp9;
            auto tmp11 = tmp0 * tmp6;
            auto tmp12 = tmp11.exp();
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = static_cast<float>(1.0507009873554805);
            auto tmp15 = at::vec::Vectorized<float>(tmp14);
            auto tmp16 = tmp4 * tmp15;
            auto tmp17 = decltype(tmp13)::blendv(tmp16, tmp13, tmp3);
            tmp17.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_elu_elu_backward_sum_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1536L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = static_cast<float>(1.0);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = static_cast<float>(1.7580993408473766);
            auto tmp9 = at::vec::Vectorized<float>(tmp8);
            auto tmp10 = tmp7 * tmp9;
            auto tmp11 = tmp0 * tmp6;
            auto tmp12 = tmp11.exp();
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = static_cast<float>(1.0507009873554805);
            auto tmp15 = at::vec::Vectorized<float>(tmp14);
            auto tmp16 = tmp4 * tmp15;
            auto tmp17 = decltype(tmp13)::blendv(tmp16, tmp13, tmp3);
            tmp17.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_sum_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1536L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
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
    buf0 = empty((4, 197951), device='cpu', dtype=torch.float32)
    cpp_fused_elu_elu_backward_0(c_void_p(addmm_5.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del addmm_5
    del tangents_1
    buf1 = empty((4, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf0, permute_6, out=buf1)
    del permute_6
    buf2 = empty((197951, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf0, (197951, 4), (1, 197951), 0), where_4, out=buf2)
    del where_4
    buf3 = empty((197951, ), device='cpu', dtype=torch.float32)
    buf4 = buf1; del buf1  # reuse
    cpp_fused_elu_elu_backward_sum_view_1(c_void_p(buf4.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf3.data_ptr()))
    del addmm_4
    del buf0
    buf5 = empty((4, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf4, permute_10, out=buf5)
    del permute_10
    buf6 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf4, (512, 4), (1, 512), 0), where_3, out=buf6)
    del where_3
    buf7 = empty((512, ), device='cpu', dtype=torch.float32)
    buf8 = buf5; del buf5  # reuse
    cpp_fused_elu_elu_backward_sum_view_2(c_void_p(buf8.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(addmm_3.data_ptr()), c_void_p(buf7.data_ptr()))
    del addmm_3
    buf9 = empty((4, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf8, permute_14, out=buf9)
    del permute_14
    buf10 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (512, 4), (1, 512), 0), clone, out=buf10)
    del clone
    buf11 = empty((512, ), device='cpu', dtype=torch.float32)
    buf12 = buf9; del buf9  # reuse
    cpp_fused_elu_elu_backward_sum_view_3(c_void_p(buf12.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(buf11.data_ptr()))
    del addmm_2
    buf13 = buf8; del buf8  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf12, permute_18, out=buf13)
    del permute_18
    buf14 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf12, (1024, 4), (1, 1024), 0), where_1, out=buf14)
    del where_1
    buf15 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf16 = buf13; del buf13  # reuse
    cpp_fused_elu_elu_backward_sum_view_4(c_void_p(buf16.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(buf15.data_ptr()))
    del addmm_1
    del buf12
    buf17 = buf4; del buf4  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf16, permute_22, out=buf17)
    del permute_22
    buf18 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf16, (512, 4), (1, 512), 0), where, out=buf18)
    del where
    buf19 = empty((512, ), device='cpu', dtype=torch.float32)
    buf20 = buf17; del buf17  # reuse
    cpp_fused_elu_elu_backward_sum_view_5(c_void_p(buf20.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(addmm.data_ptr()), c_void_p(buf19.data_ptr()))
    del addmm
    del buf16
    buf21 = empty((512, 197951), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (512, 4), (1, 512), 0), primals_13, out=buf21)
    del primals_13
    buf22 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_6(c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()))
    return (reinterpret_tensor(buf21, (512, 197951), (197951, 1), 0), reinterpret_tensor(buf18, (512, 512), (512, 1), 0), reinterpret_tensor(buf14, (1024, 512), (512, 1), 0), buf22, buf19, buf15, reinterpret_tensor(buf10, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf6, (512, 512), (512, 1), 0), reinterpret_tensor(buf2, (197951, 512), (512, 1), 0), buf11, buf7, buf3, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_13 = rand_strided((4, 197951), (197951, 1), device='cpu', dtype=torch.float32)
    addmm = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.float32)
    where = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.float32)
    where_1 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((4, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    clone = rand_strided((4, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_3 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.float32)
    where_3 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.float32)
    where_4 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_5 = rand_strided((4, 197951), (197951, 1), device='cpu', dtype=torch.float32)
    permute_6 = rand_strided((197951, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_10 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_14 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_18 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_22 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((4, 197951), (197951, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_13, addmm, where, addmm_1, where_1, addmm_2, clone, addmm_3, where_3, addmm_4, where_4, addmm_5, permute_6, permute_10, permute_14, permute_18, permute_22, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nvidia_deeprecommender', benchmark_compiled_module)
