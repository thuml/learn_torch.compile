
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


cpp_fused_sum_threshold_backward_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1000L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1000L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(2000L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3000L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
            tmp4.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_sum_threshold_backward_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(4096L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(8192L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(12288L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
            tmp4.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_sum_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(4096L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(8192L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(12288L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                tmp5.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_33, relu, relu_1, getitem, getitem_1, relu_2, relu_3, getitem_2, getitem_3, relu_4, relu_5, relu_6, getitem_4, getitem_5, relu_7, relu_8, relu_9, getitem_6, getitem_7, relu_10, relu_11, relu_12, getitem_8, getitem_9, view, clone, clone_1, permute_3, le, permute_7, le_1, permute_11, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_3, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_5, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_7, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_9, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_11, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_13, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_15, (512, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_17, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_19, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_21, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_23, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_25, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_33, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(relu, (4, 64, 224, 224), (3211264, 1, 14336, 64))
    assert_size_stride(relu_1, (4, 64, 224, 224), (3211264, 1, 14336, 64))
    assert_size_stride(getitem, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(getitem_1, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(relu_2, (4, 128, 112, 112), (1605632, 1, 14336, 128))
    assert_size_stride(relu_3, (4, 128, 112, 112), (1605632, 1, 14336, 128))
    assert_size_stride(getitem_2, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(getitem_3, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_4, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_5, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_6, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(getitem_4, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(getitem_5, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(relu_7, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_8, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_9, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(getitem_6, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_7, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_10, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_11, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_12, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_8, (4, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(getitem_9, (4, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(view, (4, 25088), (25088, 1))
    assert_size_stride(clone, (4, 4096), (4096, 1))
    assert_size_stride(clone_1, (4, 4096), (4096, 1))
    assert_size_stride(permute_3, (1000, 4096), (4096, 1))
    assert_size_stride(le, (4, 4096), (4096, 1))
    assert_size_stride(permute_7, (4096, 4096), (4096, 1))
    assert_size_stride(le_1, (4, 4096), (4096, 1))
    assert_size_stride(permute_11, (4096, 25088), (25088, 1))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_3, out=buf0)
    del permute_3
    buf1 = empty((1000, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), clone_1, out=buf1)
    del clone_1
    buf2 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf3 = buf0; del buf0  # reuse
    cpp_fused_sum_threshold_backward_view_0(c_void_p(buf3.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf2.data_ptr()))
    del le
    del tangents_1
    buf4 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf3, permute_7, out=buf4)
    del permute_7
    buf5 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (4096, 4), (1, 4096), 0), clone, out=buf5)
    del clone
    buf6 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf7 = buf4; del buf4  # reuse
    cpp_fused_sum_threshold_backward_view_1(c_void_p(buf7.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(le_1.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf3
    del le_1
    buf8 = empty((4, 25088), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf7, permute_11, out=buf8)
    del permute_11
    buf9 = empty((4096, 25088), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf7, (4096, 4), (1, 4096), 0), view, out=buf9)
    del view
    buf10 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_2(c_void_p(buf7.data_ptr()), c_void_p(buf10.data_ptr()))
    del buf7
    # Source Nodes: [], Original ATen: [aten._adaptive_avg_pool2d_backward]
    buf11 = aten._adaptive_avg_pool2d_backward(reinterpret_tensor(buf8, (4, 512, 7, 7), (25088, 49, 7, 1), 0), getitem_8)
    del buf8
    del getitem_8
    buf12 = buf11
    del buf11
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf13 = aten.max_pool2d_with_indices_backward(buf12, relu_12, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_9)
    del buf12
    del getitem_9
    buf14 = buf13
    del buf13
    buf15 = buf14; del buf14  # reuse
    cpp_fused_convolution_backward_threshold_backward_3(c_void_p(buf15.data_ptr()), c_void_p(relu_12.data_ptr()))
    del relu_12
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf16 = aten.convolution_backward(buf15, relu_11, primals_25, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf15
    del primals_25
    buf17 = buf16[0]
    buf18 = buf16[1]
    buf19 = buf16[2]
    del buf16
    buf20 = buf17; del buf17  # reuse
    cpp_fused_convolution_backward_threshold_backward_4(c_void_p(buf20.data_ptr()), c_void_p(relu_11.data_ptr()))
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf21 = aten.convolution_backward(buf20, relu_10, primals_23, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf20
    del primals_23
    buf22 = buf21[0]
    buf23 = buf21[1]
    buf24 = buf21[2]
    del buf21
    buf25 = buf22; del buf22  # reuse
    cpp_fused_convolution_backward_threshold_backward_5(c_void_p(buf25.data_ptr()), c_void_p(relu_10.data_ptr()))
    del relu_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf26 = aten.convolution_backward(buf25, getitem_6, primals_21, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf25
    del getitem_6
    del primals_21
    buf27 = buf26[0]
    buf28 = buf26[1]
    buf29 = buf26[2]
    del buf26
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf30 = aten.max_pool2d_with_indices_backward(buf27, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_7)
    del buf27
    del getitem_7
    buf31 = buf30
    del buf30
    buf32 = buf31; del buf31  # reuse
    cpp_fused_convolution_backward_threshold_backward_6(c_void_p(buf32.data_ptr()), c_void_p(relu_9.data_ptr()))
    del relu_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf33 = aten.convolution_backward(buf32, relu_8, primals_19, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf32
    del primals_19
    buf34 = buf33[0]
    buf35 = buf33[1]
    buf36 = buf33[2]
    del buf33
    buf37 = buf34; del buf34  # reuse
    cpp_fused_convolution_backward_threshold_backward_7(c_void_p(buf37.data_ptr()), c_void_p(relu_8.data_ptr()))
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf38 = aten.convolution_backward(buf37, relu_7, primals_17, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf37
    del primals_17
    buf39 = buf38[0]
    buf40 = buf38[1]
    buf41 = buf38[2]
    del buf38
    buf42 = buf39; del buf39  # reuse
    cpp_fused_convolution_backward_threshold_backward_8(c_void_p(buf42.data_ptr()), c_void_p(relu_7.data_ptr()))
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf43 = aten.convolution_backward(buf42, getitem_4, primals_15, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf42
    del getitem_4
    del primals_15
    buf44 = buf43[0]
    buf45 = buf43[1]
    buf46 = buf43[2]
    del buf43
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf47 = aten.max_pool2d_with_indices_backward(buf44, relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_5)
    del buf44
    del getitem_5
    buf48 = buf47
    del buf47
    buf49 = buf48; del buf48  # reuse
    cpp_fused_convolution_backward_threshold_backward_9(c_void_p(buf49.data_ptr()), c_void_p(relu_6.data_ptr()))
    del relu_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf50 = aten.convolution_backward(buf49, relu_5, primals_13, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf49
    del primals_13
    buf51 = buf50[0]
    buf52 = buf50[1]
    buf53 = buf50[2]
    del buf50
    buf54 = buf51; del buf51  # reuse
    cpp_fused_convolution_backward_threshold_backward_10(c_void_p(buf54.data_ptr()), c_void_p(relu_5.data_ptr()))
    del relu_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf55 = aten.convolution_backward(buf54, relu_4, primals_11, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf54
    del primals_11
    buf56 = buf55[0]
    buf57 = buf55[1]
    buf58 = buf55[2]
    del buf55
    buf59 = buf56; del buf56  # reuse
    cpp_fused_convolution_backward_threshold_backward_11(c_void_p(buf59.data_ptr()), c_void_p(relu_4.data_ptr()))
    del relu_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf60 = aten.convolution_backward(buf59, getitem_2, primals_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf59
    del getitem_2
    del primals_9
    buf61 = buf60[0]
    buf62 = buf60[1]
    buf63 = buf60[2]
    del buf60
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf64 = aten.max_pool2d_with_indices_backward(buf61, relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_3)
    del buf61
    del getitem_3
    buf65 = buf64
    del buf64
    buf66 = buf65; del buf65  # reuse
    cpp_fused_convolution_backward_threshold_backward_12(c_void_p(buf66.data_ptr()), c_void_p(relu_3.data_ptr()))
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf67 = aten.convolution_backward(buf66, relu_2, primals_7, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf66
    del primals_7
    buf68 = buf67[0]
    buf69 = buf67[1]
    buf70 = buf67[2]
    del buf67
    buf71 = buf68; del buf68  # reuse
    cpp_fused_convolution_backward_threshold_backward_13(c_void_p(buf71.data_ptr()), c_void_p(relu_2.data_ptr()))
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf72 = aten.convolution_backward(buf71, getitem, primals_5, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf71
    del getitem
    del primals_5
    buf73 = buf72[0]
    buf74 = buf72[1]
    buf75 = buf72[2]
    del buf72
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf76 = aten.max_pool2d_with_indices_backward(buf73, relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_1)
    del buf73
    del getitem_1
    buf77 = buf76
    del buf76
    buf78 = buf77; del buf77  # reuse
    cpp_fused_convolution_backward_threshold_backward_14(c_void_p(buf78.data_ptr()), c_void_p(relu_1.data_ptr()))
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf79 = aten.convolution_backward(buf78, relu, primals_3, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf78
    del primals_3
    buf80 = buf79[0]
    buf81 = buf79[1]
    buf82 = buf79[2]
    del buf79
    buf83 = buf80; del buf80  # reuse
    cpp_fused_convolution_backward_threshold_backward_15(c_void_p(buf83.data_ptr()), c_void_p(relu.data_ptr()))
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf84 = aten.convolution_backward(buf83, primals_33, primals_1, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf83
    del primals_1
    del primals_33
    buf85 = buf84[1]
    buf86 = buf84[2]
    return (buf85, buf86, buf81, buf82, buf74, buf75, buf69, buf70, buf62, buf63, buf57, buf58, buf52, buf53, buf45, buf46, buf40, buf41, buf35, buf36, buf28, buf29, buf23, buf24, buf18, buf19, reinterpret_tensor(buf9, (4096, 25088), (25088, 1), 0), buf10, reinterpret_tensor(buf5, (4096, 4096), (4096, 1), 0), buf6, reinterpret_tensor(buf1, (1000, 4096), (4096, 1), 0), buf2, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 64, 224, 224), (3211264, 1, 14336, 64), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((4, 64, 224, 224), (3211264, 1, 14336, 64), device='cpu', dtype=torch.float32)
    getitem = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.int64)
    relu_2 = rand_strided((4, 128, 112, 112), (1605632, 1, 14336, 128), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((4, 128, 112, 112), (1605632, 1, 14336, 128), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.int64)
    relu_4 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    getitem_4 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    getitem_5 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.int64)
    relu_7 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.int64)
    relu_10 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_8 = rand_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    getitem_9 = rand_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.int64)
    view = rand_strided((4, 25088), (25088, 1), device='cpu', dtype=torch.float32)
    clone = rand_strided((4, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    clone_1 = rand_strided((4, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_3 = rand_strided((1000, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((4, 4096), (4096, 1), device='cpu', dtype=torch.bool)
    permute_7 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    le_1 = rand_strided((4, 4096), (4096, 1), device='cpu', dtype=torch.bool)
    permute_11 = rand_strided((4096, 25088), (25088, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_33, relu, relu_1, getitem, getitem_1, relu_2, relu_3, getitem_2, getitem_3, relu_4, relu_5, relu_6, getitem_4, getitem_5, relu_7, relu_8, relu_9, getitem_6, getitem_7, relu_10, relu_11, relu_12, getitem_8, getitem_9, view, clone, clone_1, permute_3, le, permute_7, le_1, permute_11, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vgg16', benchmark_compiled_module)
