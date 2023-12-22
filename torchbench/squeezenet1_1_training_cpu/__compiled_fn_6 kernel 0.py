
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


cpp_fused_convolution_backward_div_threshold_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(169L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1000L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1000L*x1) + (169000L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1000L*x0)));
                        auto tmp2 = static_cast<float>(169.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (1000L*x1) + (169000L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (512L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(43264L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
            tmp7.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (512L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_threshold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(43264L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
            tmp7.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
            tmp7.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_threshold_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
            tmp7.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2916L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (256L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2916L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(93312L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
            tmp7.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2916L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (256L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2916L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_threshold_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(93312L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp6 = tmp4 + tmp5;
            auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
            tmp7.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12100L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (128L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12100L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_threshold_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(193600L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp6 = tmp4 + tmp5;
                auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                tmp7.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12100L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (128L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12100L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_threshold_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(193600L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp6 = tmp4 + tmp5;
                auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                tmp7.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3154176L); x0+=static_cast<long>(8L))
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
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, relu, getitem, getitem_1, relu_1, cat, relu_4, cat_1, getitem_2, getitem_3, relu_7, cat_2, relu_10, cat_3, getitem_4, getitem_5, relu_13, cat_4, relu_16, cat_5, relu_19, cat_6, relu_22, clone, le, le_1, le_2, le_4, le_5, le_7, le_8, le_10, le_11, le_13, le_14, le_16, le_17, le_19, le_20, le_22, le_23, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_3, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_7, (64, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_9, (16, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_11, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_13, (64, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_15, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_19, (128, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_21, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_23, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_25, (128, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_27, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_31, (192, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_33, (48, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_35, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_37, (192, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_39, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_41, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_43, (256, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_45, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_49, (256, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_51, (1000, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_53, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(relu, (4, 64, 111, 111), (788544, 1, 7104, 64))
    assert_size_stride(getitem, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(getitem_1, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(relu_1, (4, 16, 55, 55), (48400, 1, 880, 16))
    assert_size_stride(cat, (4, 128, 55, 55), (387200, 1, 7040, 128))
    assert_size_stride(relu_4, (4, 16, 55, 55), (48400, 1, 880, 16))
    assert_size_stride(cat_1, (4, 128, 55, 55), (387200, 1, 7040, 128))
    assert_size_stride(getitem_2, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(getitem_3, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(relu_7, (4, 32, 27, 27), (23328, 1, 864, 32))
    assert_size_stride(cat_2, (4, 256, 27, 27), (186624, 1, 6912, 256))
    assert_size_stride(relu_10, (4, 32, 27, 27), (23328, 1, 864, 32))
    assert_size_stride(cat_3, (4, 256, 27, 27), (186624, 1, 6912, 256))
    assert_size_stride(getitem_4, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(getitem_5, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(relu_13, (4, 48, 13, 13), (8112, 1, 624, 48))
    assert_size_stride(cat_4, (4, 384, 13, 13), (64896, 1, 4992, 384))
    assert_size_stride(relu_16, (4, 48, 13, 13), (8112, 1, 624, 48))
    assert_size_stride(cat_5, (4, 384, 13, 13), (64896, 1, 4992, 384))
    assert_size_stride(relu_19, (4, 64, 13, 13), (10816, 1, 832, 64))
    assert_size_stride(cat_6, (4, 512, 13, 13), (86528, 1, 6656, 512))
    assert_size_stride(relu_22, (4, 64, 13, 13), (10816, 1, 832, 64))
    assert_size_stride(clone, (4, 512, 13, 13), (86528, 1, 6656, 512))
    assert_size_stride(le, (4, 1000, 13, 13), (169000, 1, 13000, 1000))
    assert_size_stride(le_1, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(le_2, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(le_4, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(le_5, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(le_7, (4, 192, 13, 13), (32448, 1, 2496, 192))
    assert_size_stride(le_8, (4, 192, 13, 13), (32448, 1, 2496, 192))
    assert_size_stride(le_10, (4, 192, 13, 13), (32448, 1, 2496, 192))
    assert_size_stride(le_11, (4, 192, 13, 13), (32448, 1, 2496, 192))
    assert_size_stride(le_13, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(le_14, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(le_16, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(le_17, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(le_19, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(le_20, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(le_22, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(le_23, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty_strided((4, 1000, 13, 13), (169000, 1, 13000, 1000), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_threshold_backward_0(c_void_p(le.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del le
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.threshold_backward]
    buf1 = aten.convolution_backward(buf0, clone, primals_51, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf0
    del clone
    del primals_51
    buf2 = buf1[0]
    buf3 = buf1[1]
    buf4 = buf1[2]
    del buf1
    buf5 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_threshold_backward_1(c_void_p(le_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf5.data_ptr()))
    del le_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf6 = aten.convolution_backward(buf5, relu_22, primals_49, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_49
    buf7 = buf6[0]
    buf8 = buf6[1]
    buf9 = buf6[2]
    del buf6
    buf10 = buf5; del buf5  # reuse
    cpp_fused_convolution_backward_threshold_backward_2(c_void_p(le_2.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()))
    del buf2
    del le_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf11 = aten.convolution_backward(buf10, relu_22, primals_47, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_47
    buf12 = buf11[0]
    buf13 = buf11[1]
    buf14 = buf11[2]
    del buf11
    buf15 = buf12; del buf12  # reuse
    cpp_fused_add_convolution_backward_threshold_backward_3(c_void_p(buf15.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(buf7.data_ptr()))
    del buf7
    del relu_22
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.threshold_backward]
    buf16 = aten.convolution_backward(buf15, cat_6, primals_45, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf15
    del cat_6
    del primals_45
    buf17 = buf16[0]
    buf18 = buf16[1]
    buf19 = buf16[2]
    del buf16
    buf20 = buf10; del buf10  # reuse
    cpp_fused_convolution_backward_threshold_backward_4(c_void_p(le_4.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf20.data_ptr()))
    del le_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf21 = aten.convolution_backward(buf20, relu_19, primals_43, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_43
    buf22 = buf21[0]
    buf23 = buf21[1]
    buf24 = buf21[2]
    del buf21
    buf25 = buf20; del buf20  # reuse
    cpp_fused_convolution_backward_threshold_backward_5(c_void_p(le_5.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf25.data_ptr()))
    del buf17
    del le_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf26 = aten.convolution_backward(buf25, relu_19, primals_41, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf25
    del primals_41
    buf27 = buf26[0]
    buf28 = buf26[1]
    buf29 = buf26[2]
    del buf26
    buf30 = buf22; del buf22  # reuse
    cpp_fused_add_convolution_backward_threshold_backward_6(c_void_p(buf30.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(buf27.data_ptr()))
    del buf27
    del relu_19
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.threshold_backward]
    buf31 = aten.convolution_backward(buf30, cat_5, primals_39, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf30
    del cat_5
    del primals_39
    buf32 = buf31[0]
    buf33 = buf31[1]
    buf34 = buf31[2]
    del buf31
    buf35 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_threshold_backward_7(c_void_p(le_7.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf35.data_ptr()))
    del le_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf36 = aten.convolution_backward(buf35, relu_16, primals_37, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_37
    buf37 = buf36[0]
    buf38 = buf36[1]
    buf39 = buf36[2]
    del buf36
    buf40 = buf35; del buf35  # reuse
    cpp_fused_convolution_backward_threshold_backward_8(c_void_p(le_8.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf40.data_ptr()))
    del buf32
    del le_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf41 = aten.convolution_backward(buf40, relu_16, primals_35, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_35
    buf42 = buf41[0]
    buf43 = buf41[1]
    buf44 = buf41[2]
    del buf41
    buf45 = buf37; del buf37  # reuse
    cpp_fused_add_convolution_backward_threshold_backward_9(c_void_p(buf45.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf42.data_ptr()))
    del buf42
    del relu_16
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.threshold_backward]
    buf46 = aten.convolution_backward(buf45, cat_4, primals_33, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf45
    del cat_4
    del primals_33
    buf47 = buf46[0]
    buf48 = buf46[1]
    buf49 = buf46[2]
    del buf46
    buf50 = buf40; del buf40  # reuse
    cpp_fused_convolution_backward_threshold_backward_10(c_void_p(le_10.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf50.data_ptr()))
    del le_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf51 = aten.convolution_backward(buf50, relu_13, primals_31, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_31
    buf52 = buf51[0]
    buf53 = buf51[1]
    buf54 = buf51[2]
    del buf51
    buf55 = buf50; del buf50  # reuse
    cpp_fused_convolution_backward_threshold_backward_11(c_void_p(le_11.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf55.data_ptr()))
    del buf47
    del le_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf56 = aten.convolution_backward(buf55, relu_13, primals_29, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf55
    del primals_29
    buf57 = buf56[0]
    buf58 = buf56[1]
    buf59 = buf56[2]
    del buf56
    buf60 = buf52; del buf52  # reuse
    cpp_fused_add_convolution_backward_threshold_backward_12(c_void_p(buf60.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(buf57.data_ptr()))
    del buf57
    del relu_13
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.threshold_backward]
    buf61 = aten.convolution_backward(buf60, getitem_4, primals_27, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf60
    del getitem_4
    del primals_27
    buf62 = buf61[0]
    buf63 = buf61[1]
    buf64 = buf61[2]
    del buf61
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf65 = aten.max_pool2d_with_indices_backward(buf62, cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_5)
    del buf62
    del cat_3
    del getitem_5
    buf66 = buf65
    del buf65
    buf67 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_threshold_backward_13(c_void_p(le_13.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    del le_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf68 = aten.convolution_backward(buf67, relu_10, primals_25, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_25
    buf69 = buf68[0]
    buf70 = buf68[1]
    buf71 = buf68[2]
    del buf68
    buf72 = buf67; del buf67  # reuse
    cpp_fused_convolution_backward_threshold_backward_14(c_void_p(le_14.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf72.data_ptr()))
    del buf66
    del le_14
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf73 = aten.convolution_backward(buf72, relu_10, primals_23, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_23
    buf74 = buf73[0]
    buf75 = buf73[1]
    buf76 = buf73[2]
    del buf73
    buf77 = buf69; del buf69  # reuse
    cpp_fused_add_convolution_backward_threshold_backward_15(c_void_p(buf77.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(buf74.data_ptr()))
    del buf74
    del relu_10
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.threshold_backward]
    buf78 = aten.convolution_backward(buf77, cat_2, primals_21, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf77
    del cat_2
    del primals_21
    buf79 = buf78[0]
    buf80 = buf78[1]
    buf81 = buf78[2]
    del buf78
    buf82 = buf72; del buf72  # reuse
    cpp_fused_convolution_backward_threshold_backward_16(c_void_p(le_16.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf82.data_ptr()))
    del le_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf83 = aten.convolution_backward(buf82, relu_7, primals_19, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_19
    buf84 = buf83[0]
    buf85 = buf83[1]
    buf86 = buf83[2]
    del buf83
    buf87 = buf82; del buf82  # reuse
    cpp_fused_convolution_backward_threshold_backward_17(c_void_p(le_17.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf87.data_ptr()))
    del buf79
    del le_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf88 = aten.convolution_backward(buf87, relu_7, primals_17, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf87
    del primals_17
    buf89 = buf88[0]
    buf90 = buf88[1]
    buf91 = buf88[2]
    del buf88
    buf92 = buf84; del buf84  # reuse
    cpp_fused_add_convolution_backward_threshold_backward_18(c_void_p(buf92.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(buf89.data_ptr()))
    del buf89
    del relu_7
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.threshold_backward]
    buf93 = aten.convolution_backward(buf92, getitem_2, primals_15, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf92
    del getitem_2
    del primals_15
    buf94 = buf93[0]
    buf95 = buf93[1]
    buf96 = buf93[2]
    del buf93
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf97 = aten.max_pool2d_with_indices_backward(buf94, cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_3)
    del buf94
    del cat_1
    del getitem_3
    buf98 = buf97
    del buf97
    buf99 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_threshold_backward_19(c_void_p(le_19.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    del le_19
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf100 = aten.convolution_backward(buf99, relu_4, primals_13, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_13
    buf101 = buf100[0]
    buf102 = buf100[1]
    buf103 = buf100[2]
    del buf100
    buf104 = buf99; del buf99  # reuse
    cpp_fused_convolution_backward_threshold_backward_20(c_void_p(le_20.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf104.data_ptr()))
    del buf98
    del le_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf105 = aten.convolution_backward(buf104, relu_4, primals_11, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_11
    buf106 = buf105[0]
    buf107 = buf105[1]
    buf108 = buf105[2]
    del buf105
    buf109 = buf101; del buf101  # reuse
    cpp_fused_add_convolution_backward_threshold_backward_21(c_void_p(buf109.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf106.data_ptr()))
    del buf106
    del relu_4
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.threshold_backward]
    buf110 = aten.convolution_backward(buf109, cat, primals_9, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf109
    del cat
    del primals_9
    buf111 = buf110[0]
    buf112 = buf110[1]
    buf113 = buf110[2]
    del buf110
    buf114 = buf104; del buf104  # reuse
    cpp_fused_convolution_backward_threshold_backward_22(c_void_p(le_22.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf114.data_ptr()))
    del le_22
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf115 = aten.convolution_backward(buf114, relu_1, primals_7, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_7
    buf116 = buf115[0]
    buf117 = buf115[1]
    buf118 = buf115[2]
    del buf115
    buf119 = buf114; del buf114  # reuse
    cpp_fused_convolution_backward_threshold_backward_23(c_void_p(le_23.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf119.data_ptr()))
    del buf111
    del le_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf120 = aten.convolution_backward(buf119, relu_1, primals_5, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf119
    del primals_5
    buf121 = buf120[0]
    buf122 = buf120[1]
    buf123 = buf120[2]
    del buf120
    buf124 = buf116; del buf116  # reuse
    cpp_fused_add_convolution_backward_threshold_backward_24(c_void_p(buf124.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(buf121.data_ptr()))
    del buf121
    del relu_1
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.threshold_backward]
    buf125 = aten.convolution_backward(buf124, getitem, primals_3, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf124
    del getitem
    del primals_3
    buf126 = buf125[0]
    buf127 = buf125[1]
    buf128 = buf125[2]
    del buf125
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf129 = aten.max_pool2d_with_indices_backward(buf126, relu, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_1)
    del buf126
    del getitem_1
    buf130 = buf129
    del buf129
    buf131 = buf130; del buf130  # reuse
    cpp_fused_convolution_backward_threshold_backward_25(c_void_p(buf131.data_ptr()), c_void_p(relu.data_ptr()))
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf132 = aten.convolution_backward(buf131, primals_53, primals_1, [64], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf131
    del primals_1
    del primals_53
    buf133 = buf132[1]
    buf134 = buf132[2]
    return (buf133, buf134, buf127, buf128, buf122, buf123, buf117, buf118, buf112, buf113, buf107, buf108, buf102, buf103, buf95, buf96, buf90, buf91, buf85, buf86, buf80, buf81, buf75, buf76, buf70, buf71, buf63, buf64, buf58, buf59, buf53, buf54, buf48, buf49, buf43, buf44, buf38, buf39, buf33, buf34, buf28, buf29, buf23, buf24, buf18, buf19, buf13, buf14, buf8, buf9, buf3, buf4, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((16, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((192, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((48, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((192, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((1000, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 64, 111, 111), (788544, 1, 7104, 64), device='cpu', dtype=torch.float32)
    getitem = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.int64)
    relu_1 = rand_strided((4, 16, 55, 55), (48400, 1, 880, 16), device='cpu', dtype=torch.float32)
    cat = rand_strided((4, 128, 55, 55), (387200, 1, 7040, 128), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 16, 55, 55), (48400, 1, 880, 16), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((4, 128, 55, 55), (387200, 1, 7040, 128), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.int64)
    relu_7 = rand_strided((4, 32, 27, 27), (23328, 1, 864, 32), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((4, 256, 27, 27), (186624, 1, 6912, 256), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((4, 32, 27, 27), (23328, 1, 864, 32), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((4, 256, 27, 27), (186624, 1, 6912, 256), device='cpu', dtype=torch.float32)
    getitem_4 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.float32)
    getitem_5 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.int64)
    relu_13 = rand_strided((4, 48, 13, 13), (8112, 1, 624, 48), device='cpu', dtype=torch.float32)
    cat_4 = rand_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((4, 48, 13, 13), (8112, 1, 624, 48), device='cpu', dtype=torch.float32)
    cat_5 = rand_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((4, 64, 13, 13), (10816, 1, 832, 64), device='cpu', dtype=torch.float32)
    cat_6 = rand_strided((4, 512, 13, 13), (86528, 1, 6656, 512), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((4, 64, 13, 13), (10816, 1, 832, 64), device='cpu', dtype=torch.float32)
    clone = rand_strided((4, 512, 13, 13), (86528, 1, 6656, 512), device='cpu', dtype=torch.float32)
    le = rand_strided((4, 1000, 13, 13), (169000, 1, 13000, 1000), device='cpu', dtype=torch.bool)
    le_1 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.bool)
    le_2 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.bool)
    le_4 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.bool)
    le_5 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.bool)
    le_7 = rand_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.bool)
    le_8 = rand_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.bool)
    le_10 = rand_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.bool)
    le_11 = rand_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.bool)
    le_13 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.bool)
    le_14 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.bool)
    le_16 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.bool)
    le_17 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.bool)
    le_19 = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.bool)
    le_20 = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.bool)
    le_22 = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.bool)
    le_23 = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, relu, getitem, getitem_1, relu_1, cat, relu_4, cat_1, getitem_2, getitem_3, relu_7, cat_2, relu_10, cat_3, getitem_4, getitem_5, relu_13, cat_4, relu_16, cat_5, relu_19, cat_6, relu_22, clone, le, le_1, le_2, le_4, le_5, le_7, le_8, le_10, le_11, le_13, le_14, le_16, le_17, le_19, le_20, le_22, le_23, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('squeezenet1_1', benchmark_compiled_module)
