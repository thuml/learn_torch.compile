
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


cpp_fused_convolution_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (256L*x1) + (768L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_addcmul_addmm_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_addmm_mul_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2)];
                        auto tmp2 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2)];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    tmp7.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x0) + (75264L*(c10::div_floor_integer((x1 + x1_inner), 384L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_clone_mul_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp5 + tmp9;
                            auto tmp11 = tmp4 * tmp10;
                            auto tmp12 = tmp0 + tmp11;
                            tmp12.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2)];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                        auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_addcmul_mean_mul_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp7[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x2 + (196L*x1) + (75264L*x0)), static_cast<long>(196L), tmp7, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (384L*x2_inner) + (75264L*x0)));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (384L*x2) + (384L*x2_inner) + (75264L*x0)));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp5 + tmp9;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp14 = tmp10 + tmp13;
                                auto tmp15 = tmp4 * tmp14;
                                auto tmp16 = tmp0 + tmp15;
                                tmp_acc0_vec = tmp_acc0_vec + tmp16;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = tmp9 + tmp12;
                            auto tmp14 = tmp4 * tmp13;
                            auto tmp15 = tmp0 + tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1 = args
    args.clear()
    assert_size_stride(arg0_1, (384, ), (1, ))
    assert_size_stride(arg1_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg2_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg3_1, (384, ), (1, ))
    assert_size_stride(arg4_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg5_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg6_1, (384, ), (1, ))
    assert_size_stride(arg7_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg8_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg9_1, (384, ), (1, ))
    assert_size_stride(arg10_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg11_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg12_1, (384, ), (1, ))
    assert_size_stride(arg13_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg14_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg15_1, (384, ), (1, ))
    assert_size_stride(arg16_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg17_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg18_1, (384, ), (1, ))
    assert_size_stride(arg19_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg20_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg21_1, (384, ), (1, ))
    assert_size_stride(arg22_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg23_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg24_1, (384, ), (1, ))
    assert_size_stride(arg25_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg26_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg27_1, (384, ), (1, ))
    assert_size_stride(arg28_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg29_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg30_1, (384, ), (1, ))
    assert_size_stride(arg31_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg32_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg33_1, (384, ), (1, ))
    assert_size_stride(arg34_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg35_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg36_1, (384, ), (1, ))
    assert_size_stride(arg37_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg38_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg41_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg42_1, (384, ), (1, ))
    assert_size_stride(arg43_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg44_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg45_1, (384, ), (1, ))
    assert_size_stride(arg46_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg47_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg48_1, (384, ), (1, ))
    assert_size_stride(arg49_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg50_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg51_1, (384, ), (1, ))
    assert_size_stride(arg52_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg53_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg56_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg59_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg60_1, (384, ), (1, ))
    assert_size_stride(arg61_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg62_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg65_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg68_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg69_1, (384, ), (1, ))
    assert_size_stride(arg70_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg71_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg72_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg73_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg74_1, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (196, 196), (196, 1))
    assert_size_stride(arg77_1, (196, ), (1, ))
    assert_size_stride(arg78_1, (1536, 384), (384, 1))
    assert_size_stride(arg79_1, (1536, ), (1, ))
    assert_size_stride(arg80_1, (384, 1536), (1536, 1))
    assert_size_stride(arg81_1, (384, ), (1, ))
    assert_size_stride(arg82_1, (196, 196), (196, 1))
    assert_size_stride(arg83_1, (196, ), (1, ))
    assert_size_stride(arg84_1, (1536, 384), (384, 1))
    assert_size_stride(arg85_1, (1536, ), (1, ))
    assert_size_stride(arg86_1, (384, 1536), (1536, 1))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (196, 196), (196, 1))
    assert_size_stride(arg89_1, (196, ), (1, ))
    assert_size_stride(arg90_1, (1536, 384), (384, 1))
    assert_size_stride(arg91_1, (1536, ), (1, ))
    assert_size_stride(arg92_1, (384, 1536), (1536, 1))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (196, 196), (196, 1))
    assert_size_stride(arg95_1, (196, ), (1, ))
    assert_size_stride(arg96_1, (1536, 384), (384, 1))
    assert_size_stride(arg97_1, (1536, ), (1, ))
    assert_size_stride(arg98_1, (384, 1536), (1536, 1))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (196, 196), (196, 1))
    assert_size_stride(arg101_1, (196, ), (1, ))
    assert_size_stride(arg102_1, (1536, 384), (384, 1))
    assert_size_stride(arg103_1, (1536, ), (1, ))
    assert_size_stride(arg104_1, (384, 1536), (1536, 1))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (196, 196), (196, 1))
    assert_size_stride(arg107_1, (196, ), (1, ))
    assert_size_stride(arg108_1, (1536, 384), (384, 1))
    assert_size_stride(arg109_1, (1536, ), (1, ))
    assert_size_stride(arg110_1, (384, 1536), (1536, 1))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (196, 196), (196, 1))
    assert_size_stride(arg113_1, (196, ), (1, ))
    assert_size_stride(arg114_1, (1536, 384), (384, 1))
    assert_size_stride(arg115_1, (1536, ), (1, ))
    assert_size_stride(arg116_1, (384, 1536), (1536, 1))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (196, 196), (196, 1))
    assert_size_stride(arg119_1, (196, ), (1, ))
    assert_size_stride(arg120_1, (1536, 384), (384, 1))
    assert_size_stride(arg121_1, (1536, ), (1, ))
    assert_size_stride(arg122_1, (384, 1536), (1536, 1))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (196, 196), (196, 1))
    assert_size_stride(arg125_1, (196, ), (1, ))
    assert_size_stride(arg126_1, (1536, 384), (384, 1))
    assert_size_stride(arg127_1, (1536, ), (1, ))
    assert_size_stride(arg128_1, (384, 1536), (1536, 1))
    assert_size_stride(arg129_1, (384, ), (1, ))
    assert_size_stride(arg130_1, (196, 196), (196, 1))
    assert_size_stride(arg131_1, (196, ), (1, ))
    assert_size_stride(arg132_1, (1536, 384), (384, 1))
    assert_size_stride(arg133_1, (1536, ), (1, ))
    assert_size_stride(arg134_1, (384, 1536), (1536, 1))
    assert_size_stride(arg135_1, (384, ), (1, ))
    assert_size_stride(arg136_1, (196, 196), (196, 1))
    assert_size_stride(arg137_1, (196, ), (1, ))
    assert_size_stride(arg138_1, (1536, 384), (384, 1))
    assert_size_stride(arg139_1, (1536, ), (1, ))
    assert_size_stride(arg140_1, (384, 1536), (1536, 1))
    assert_size_stride(arg141_1, (384, ), (1, ))
    assert_size_stride(arg142_1, (196, 196), (196, 1))
    assert_size_stride(arg143_1, (196, ), (1, ))
    assert_size_stride(arg144_1, (1536, 384), (384, 1))
    assert_size_stride(arg145_1, (1536, ), (1, ))
    assert_size_stride(arg146_1, (384, 1536), (1536, 1))
    assert_size_stride(arg147_1, (384, ), (1, ))
    assert_size_stride(arg148_1, (1000, 384), (384, 1))
    assert_size_stride(arg149_1, (1000, ), (1, ))
    assert_size_stride(arg150_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((384, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg150_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg150_1
    del arg74_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg75_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg75_1
    del buf0
    del buf1
    buf3 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((3072, 196), (1, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_addcmul_addmm_1(c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg1_1
    del arg2_1
    buf5 = reinterpret_tensor(buf3, (3072, 196), (196, 1), 0); del buf3  # reuse
    # Source Nodes: [getattr_l__mod___blocks___0___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg77_1, buf4, reinterpret_tensor(arg76_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf5)
    del arg76_1
    del arg77_1
    buf6 = reinterpret_tensor(buf4, (8, 196, 384), (75264, 384, 1), 0); del buf4  # reuse
    cpp_fused_add_addcmul_clone_mul_2(c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg4_1
    del arg5_1
    buf7 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf6, (1568, 384), (384, 1), 0), reinterpret_tensor(arg78_1, (384, 1536), (1, 384), 0), out=buf7)
    del arg78_1
    buf8 = reinterpret_tensor(buf7, (8, 196, 1536), (301056, 1536, 1), 0); del buf7  # reuse
    cpp_fused_add_gelu_3(c_void_p(buf8.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg79_1
    buf9 = reinterpret_tensor(buf6, (1568, 384), (384, 1), 0); del buf6  # reuse
    # Source Nodes: [x_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg81_1, reinterpret_tensor(buf8, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg80_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf9)
    del arg80_1
    del arg81_1
    buf10 = reinterpret_tensor(buf2, (8, 196, 384), (75264, 384, 1), 0); del buf2  # reuse
    buf11 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((3072, 196), (1, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_add_addcmul_addmm_mul_4(c_void_p(buf10.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del arg0_1
    del arg3_1
    del arg7_1
    del arg8_1
    buf13 = reinterpret_tensor(buf9, (3072, 196), (196, 1), 0); del buf9  # reuse
    # Source Nodes: [getattr_l__mod___blocks___1___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg83_1, buf12, reinterpret_tensor(arg82_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf13)
    del arg82_1
    del arg83_1
    buf14 = reinterpret_tensor(buf12, (8, 196, 384), (75264, 384, 1), 0); del buf12  # reuse
    cpp_fused_add_addcmul_clone_mul_5(c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()))
    del arg10_1
    del arg11_1
    buf15 = reinterpret_tensor(buf8, (1568, 1536), (1536, 1), 0); del buf8  # reuse
    # Source Nodes: [x_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (1568, 384), (384, 1), 0), reinterpret_tensor(arg84_1, (384, 1536), (1, 384), 0), out=buf15)
    del arg84_1
    buf16 = reinterpret_tensor(buf15, (8, 196, 1536), (301056, 1536, 1), 0); del buf15  # reuse
    cpp_fused_add_gelu_6(c_void_p(buf16.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg85_1
    buf17 = reinterpret_tensor(buf14, (1568, 384), (384, 1), 0); del buf14  # reuse
    # Source Nodes: [x_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf16, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg86_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf17)
    del arg86_1
    del arg87_1
    buf18 = buf10; del buf10  # reuse
    buf19 = reinterpret_tensor(buf5, (8, 196, 384), (75264, 384, 1), 0); del buf5  # reuse
    buf20 = reinterpret_tensor(buf11, (3072, 196), (1, 3072), 0); del buf11  # reuse
    cpp_fused_add_addcmul_addmm_mul_7(c_void_p(buf18.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    del arg13_1
    del arg14_1
    del arg6_1
    del arg9_1
    buf21 = reinterpret_tensor(buf19, (3072, 196), (196, 1), 0); del buf19  # reuse
    # Source Nodes: [getattr_l__mod___blocks___2___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg89_1, buf20, reinterpret_tensor(arg88_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf21)
    del arg88_1
    del arg89_1
    buf22 = reinterpret_tensor(buf20, (8, 196, 384), (75264, 384, 1), 0); del buf20  # reuse
    cpp_fused_add_addcmul_clone_mul_8(c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del arg16_1
    del arg17_1
    buf23 = reinterpret_tensor(buf16, (1568, 1536), (1536, 1), 0); del buf16  # reuse
    # Source Nodes: [x_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (1568, 384), (384, 1), 0), reinterpret_tensor(arg90_1, (384, 1536), (1, 384), 0), out=buf23)
    del arg90_1
    buf24 = reinterpret_tensor(buf23, (8, 196, 1536), (301056, 1536, 1), 0); del buf23  # reuse
    cpp_fused_add_gelu_9(c_void_p(buf24.data_ptr()), c_void_p(arg91_1.data_ptr()))
    del arg91_1
    buf25 = reinterpret_tensor(buf22, (1568, 384), (384, 1), 0); del buf22  # reuse
    # Source Nodes: [x_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg93_1, reinterpret_tensor(buf24, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg92_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf25)
    del arg92_1
    del arg93_1
    buf26 = buf18; del buf18  # reuse
    buf27 = reinterpret_tensor(buf17, (8, 196, 384), (75264, 384, 1), 0); del buf17  # reuse
    buf28 = reinterpret_tensor(buf13, (3072, 196), (1, 3072), 0); del buf13  # reuse
    cpp_fused_add_addcmul_addmm_mul_10(c_void_p(buf26.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    del arg12_1
    del arg15_1
    del arg19_1
    del arg20_1
    buf29 = reinterpret_tensor(buf27, (3072, 196), (196, 1), 0); del buf27  # reuse
    # Source Nodes: [getattr_l__mod___blocks___3___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, buf28, reinterpret_tensor(arg94_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf29)
    del arg94_1
    del arg95_1
    buf30 = reinterpret_tensor(buf28, (8, 196, 384), (75264, 384, 1), 0); del buf28  # reuse
    cpp_fused_add_addcmul_clone_mul_11(c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    del arg22_1
    del arg23_1
    buf31 = reinterpret_tensor(buf24, (1568, 1536), (1536, 1), 0); del buf24  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (1568, 384), (384, 1), 0), reinterpret_tensor(arg96_1, (384, 1536), (1, 384), 0), out=buf31)
    del arg96_1
    buf32 = reinterpret_tensor(buf31, (8, 196, 1536), (301056, 1536, 1), 0); del buf31  # reuse
    cpp_fused_add_gelu_12(c_void_p(buf32.data_ptr()), c_void_p(arg97_1.data_ptr()))
    del arg97_1
    buf33 = reinterpret_tensor(buf30, (1568, 384), (384, 1), 0); del buf30  # reuse
    # Source Nodes: [x_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg99_1, reinterpret_tensor(buf32, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg98_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf33)
    del arg98_1
    del arg99_1
    buf34 = buf26; del buf26  # reuse
    buf35 = reinterpret_tensor(buf25, (8, 196, 384), (75264, 384, 1), 0); del buf25  # reuse
    buf36 = reinterpret_tensor(buf21, (3072, 196), (1, 3072), 0); del buf21  # reuse
    cpp_fused_add_addcmul_addmm_mul_13(c_void_p(buf34.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del arg18_1
    del arg21_1
    del arg25_1
    del arg26_1
    buf37 = reinterpret_tensor(buf35, (3072, 196), (196, 1), 0); del buf35  # reuse
    # Source Nodes: [getattr_l__mod___blocks___4___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, buf36, reinterpret_tensor(arg100_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf37)
    del arg100_1
    del arg101_1
    buf38 = reinterpret_tensor(buf36, (8, 196, 384), (75264, 384, 1), 0); del buf36  # reuse
    cpp_fused_add_addcmul_clone_mul_14(c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    del arg28_1
    del arg29_1
    buf39 = reinterpret_tensor(buf32, (1568, 1536), (1536, 1), 0); del buf32  # reuse
    # Source Nodes: [x_37], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf38, (1568, 384), (384, 1), 0), reinterpret_tensor(arg102_1, (384, 1536), (1, 384), 0), out=buf39)
    del arg102_1
    buf40 = reinterpret_tensor(buf39, (8, 196, 1536), (301056, 1536, 1), 0); del buf39  # reuse
    cpp_fused_add_gelu_15(c_void_p(buf40.data_ptr()), c_void_p(arg103_1.data_ptr()))
    del arg103_1
    buf41 = reinterpret_tensor(buf38, (1568, 384), (384, 1), 0); del buf38  # reuse
    # Source Nodes: [x_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf40, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg104_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf41)
    del arg104_1
    del arg105_1
    buf42 = buf34; del buf34  # reuse
    buf43 = reinterpret_tensor(buf33, (8, 196, 384), (75264, 384, 1), 0); del buf33  # reuse
    buf44 = reinterpret_tensor(buf29, (3072, 196), (1, 3072), 0); del buf29  # reuse
    cpp_fused_add_addcmul_addmm_mul_16(c_void_p(buf42.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    del arg24_1
    del arg27_1
    del arg31_1
    del arg32_1
    buf45 = reinterpret_tensor(buf43, (3072, 196), (196, 1), 0); del buf43  # reuse
    # Source Nodes: [getattr_l__mod___blocks___5___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg107_1, buf44, reinterpret_tensor(arg106_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf45)
    del arg106_1
    del arg107_1
    buf46 = reinterpret_tensor(buf44, (8, 196, 384), (75264, 384, 1), 0); del buf44  # reuse
    cpp_fused_add_addcmul_clone_mul_17(c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    del arg34_1
    del arg35_1
    buf47 = reinterpret_tensor(buf40, (1568, 1536), (1536, 1), 0); del buf40  # reuse
    # Source Nodes: [x_45], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf46, (1568, 384), (384, 1), 0), reinterpret_tensor(arg108_1, (384, 1536), (1, 384), 0), out=buf47)
    del arg108_1
    buf48 = reinterpret_tensor(buf47, (8, 196, 1536), (301056, 1536, 1), 0); del buf47  # reuse
    cpp_fused_add_gelu_18(c_void_p(buf48.data_ptr()), c_void_p(arg109_1.data_ptr()))
    del arg109_1
    buf49 = reinterpret_tensor(buf46, (1568, 384), (384, 1), 0); del buf46  # reuse
    # Source Nodes: [x_49], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg111_1, reinterpret_tensor(buf48, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg110_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf49)
    del arg110_1
    del arg111_1
    buf50 = buf42; del buf42  # reuse
    buf51 = reinterpret_tensor(buf41, (8, 196, 384), (75264, 384, 1), 0); del buf41  # reuse
    buf52 = reinterpret_tensor(buf37, (3072, 196), (1, 3072), 0); del buf37  # reuse
    cpp_fused_add_addcmul_addmm_mul_19(c_void_p(buf50.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    del arg30_1
    del arg33_1
    del arg37_1
    del arg38_1
    buf53 = reinterpret_tensor(buf51, (3072, 196), (196, 1), 0); del buf51  # reuse
    # Source Nodes: [getattr_l__mod___blocks___6___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg113_1, buf52, reinterpret_tensor(arg112_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf53)
    del arg112_1
    del arg113_1
    buf54 = reinterpret_tensor(buf52, (8, 196, 384), (75264, 384, 1), 0); del buf52  # reuse
    cpp_fused_add_addcmul_clone_mul_20(c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del arg40_1
    del arg41_1
    buf55 = reinterpret_tensor(buf48, (1568, 1536), (1536, 1), 0); del buf48  # reuse
    # Source Nodes: [x_53], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf54, (1568, 384), (384, 1), 0), reinterpret_tensor(arg114_1, (384, 1536), (1, 384), 0), out=buf55)
    del arg114_1
    buf56 = reinterpret_tensor(buf55, (8, 196, 1536), (301056, 1536, 1), 0); del buf55  # reuse
    cpp_fused_add_gelu_21(c_void_p(buf56.data_ptr()), c_void_p(arg115_1.data_ptr()))
    del arg115_1
    buf57 = reinterpret_tensor(buf54, (1568, 384), (384, 1), 0); del buf54  # reuse
    # Source Nodes: [x_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg117_1, reinterpret_tensor(buf56, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg116_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf57)
    del arg116_1
    del arg117_1
    buf58 = buf50; del buf50  # reuse
    buf59 = reinterpret_tensor(buf49, (8, 196, 384), (75264, 384, 1), 0); del buf49  # reuse
    buf60 = reinterpret_tensor(buf45, (3072, 196), (1, 3072), 0); del buf45  # reuse
    cpp_fused_add_addcmul_addmm_mul_22(c_void_p(buf58.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    del arg36_1
    del arg39_1
    del arg43_1
    del arg44_1
    buf61 = reinterpret_tensor(buf59, (3072, 196), (196, 1), 0); del buf59  # reuse
    # Source Nodes: [getattr_l__mod___blocks___7___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, buf60, reinterpret_tensor(arg118_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf61)
    del arg118_1
    del arg119_1
    buf62 = reinterpret_tensor(buf60, (8, 196, 384), (75264, 384, 1), 0); del buf60  # reuse
    cpp_fused_add_addcmul_clone_mul_23(c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del arg46_1
    del arg47_1
    buf63 = reinterpret_tensor(buf56, (1568, 1536), (1536, 1), 0); del buf56  # reuse
    # Source Nodes: [x_61], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (1568, 384), (384, 1), 0), reinterpret_tensor(arg120_1, (384, 1536), (1, 384), 0), out=buf63)
    del arg120_1
    buf64 = reinterpret_tensor(buf63, (8, 196, 1536), (301056, 1536, 1), 0); del buf63  # reuse
    cpp_fused_add_gelu_24(c_void_p(buf64.data_ptr()), c_void_p(arg121_1.data_ptr()))
    del arg121_1
    buf65 = reinterpret_tensor(buf62, (1568, 384), (384, 1), 0); del buf62  # reuse
    # Source Nodes: [x_65], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg123_1, reinterpret_tensor(buf64, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg122_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf65)
    del arg122_1
    del arg123_1
    buf66 = buf58; del buf58  # reuse
    buf67 = reinterpret_tensor(buf57, (8, 196, 384), (75264, 384, 1), 0); del buf57  # reuse
    buf68 = reinterpret_tensor(buf53, (3072, 196), (1, 3072), 0); del buf53  # reuse
    cpp_fused_add_addcmul_addmm_mul_25(c_void_p(buf66.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del arg42_1
    del arg45_1
    del arg49_1
    del arg50_1
    buf69 = reinterpret_tensor(buf67, (3072, 196), (196, 1), 0); del buf67  # reuse
    # Source Nodes: [getattr_l__mod___blocks___8___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg125_1, buf68, reinterpret_tensor(arg124_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf69)
    del arg124_1
    del arg125_1
    buf70 = reinterpret_tensor(buf68, (8, 196, 384), (75264, 384, 1), 0); del buf68  # reuse
    cpp_fused_add_addcmul_clone_mul_26(c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del arg52_1
    del arg53_1
    buf71 = reinterpret_tensor(buf64, (1568, 1536), (1536, 1), 0); del buf64  # reuse
    # Source Nodes: [x_69], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (1568, 384), (384, 1), 0), reinterpret_tensor(arg126_1, (384, 1536), (1, 384), 0), out=buf71)
    del arg126_1
    buf72 = reinterpret_tensor(buf71, (8, 196, 1536), (301056, 1536, 1), 0); del buf71  # reuse
    cpp_fused_add_gelu_27(c_void_p(buf72.data_ptr()), c_void_p(arg127_1.data_ptr()))
    del arg127_1
    buf73 = reinterpret_tensor(buf70, (1568, 384), (384, 1), 0); del buf70  # reuse
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf72, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg128_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf73)
    del arg128_1
    del arg129_1
    buf74 = buf66; del buf66  # reuse
    buf75 = reinterpret_tensor(buf65, (8, 196, 384), (75264, 384, 1), 0); del buf65  # reuse
    buf76 = reinterpret_tensor(buf61, (3072, 196), (1, 3072), 0); del buf61  # reuse
    cpp_fused_add_addcmul_addmm_mul_28(c_void_p(buf74.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    del arg48_1
    del arg51_1
    del arg55_1
    del arg56_1
    buf77 = reinterpret_tensor(buf75, (3072, 196), (196, 1), 0); del buf75  # reuse
    # Source Nodes: [getattr_l__mod___blocks___9___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, buf76, reinterpret_tensor(arg130_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf77)
    del arg130_1
    del arg131_1
    buf78 = reinterpret_tensor(buf76, (8, 196, 384), (75264, 384, 1), 0); del buf76  # reuse
    cpp_fused_add_addcmul_clone_mul_29(c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg58_1
    del arg59_1
    buf79 = reinterpret_tensor(buf72, (1568, 1536), (1536, 1), 0); del buf72  # reuse
    # Source Nodes: [x_77], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf78, (1568, 384), (384, 1), 0), reinterpret_tensor(arg132_1, (384, 1536), (1, 384), 0), out=buf79)
    del arg132_1
    buf80 = reinterpret_tensor(buf79, (8, 196, 1536), (301056, 1536, 1), 0); del buf79  # reuse
    cpp_fused_add_gelu_30(c_void_p(buf80.data_ptr()), c_void_p(arg133_1.data_ptr()))
    del arg133_1
    buf81 = reinterpret_tensor(buf78, (1568, 384), (384, 1), 0); del buf78  # reuse
    # Source Nodes: [x_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg135_1, reinterpret_tensor(buf80, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg134_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf81)
    del arg134_1
    del arg135_1
    buf82 = buf74; del buf74  # reuse
    buf83 = reinterpret_tensor(buf73, (8, 196, 384), (75264, 384, 1), 0); del buf73  # reuse
    buf84 = reinterpret_tensor(buf69, (3072, 196), (1, 3072), 0); del buf69  # reuse
    cpp_fused_add_addcmul_addmm_mul_31(c_void_p(buf82.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    del arg54_1
    del arg57_1
    del arg61_1
    del arg62_1
    buf85 = reinterpret_tensor(buf83, (3072, 196), (196, 1), 0); del buf83  # reuse
    # Source Nodes: [getattr_l__mod___blocks___10___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg137_1, buf84, reinterpret_tensor(arg136_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf85)
    del arg136_1
    del arg137_1
    buf86 = reinterpret_tensor(buf84, (8, 196, 384), (75264, 384, 1), 0); del buf84  # reuse
    cpp_fused_add_addcmul_clone_mul_32(c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    del arg64_1
    del arg65_1
    buf87 = reinterpret_tensor(buf80, (1568, 1536), (1536, 1), 0); del buf80  # reuse
    # Source Nodes: [x_85], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (1568, 384), (384, 1), 0), reinterpret_tensor(arg138_1, (384, 1536), (1, 384), 0), out=buf87)
    del arg138_1
    buf88 = reinterpret_tensor(buf87, (8, 196, 1536), (301056, 1536, 1), 0); del buf87  # reuse
    cpp_fused_add_gelu_33(c_void_p(buf88.data_ptr()), c_void_p(arg139_1.data_ptr()))
    del arg139_1
    buf89 = reinterpret_tensor(buf86, (1568, 384), (384, 1), 0); del buf86  # reuse
    # Source Nodes: [x_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg141_1, reinterpret_tensor(buf88, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg140_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf89)
    del arg140_1
    del arg141_1
    buf90 = buf82; del buf82  # reuse
    buf91 = reinterpret_tensor(buf81, (8, 196, 384), (75264, 384, 1), 0); del buf81  # reuse
    buf92 = reinterpret_tensor(buf77, (3072, 196), (1, 3072), 0); del buf77  # reuse
    cpp_fused_add_addcmul_addmm_mul_34(c_void_p(buf90.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    del arg60_1
    del arg63_1
    del arg67_1
    del arg68_1
    del buf85
    del buf89
    buf93 = reinterpret_tensor(buf91, (3072, 196), (196, 1), 0); del buf91  # reuse
    # Source Nodes: [getattr_l__mod___blocks___11___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, buf92, reinterpret_tensor(arg142_1, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf93)
    del arg142_1
    del arg143_1
    buf94 = reinterpret_tensor(buf92, (8, 196, 384), (75264, 384, 1), 0); del buf92  # reuse
    cpp_fused_add_addcmul_clone_mul_35(c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del arg70_1
    del arg71_1
    buf95 = reinterpret_tensor(buf88, (1568, 1536), (1536, 1), 0); del buf88  # reuse
    # Source Nodes: [x_93], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (1568, 384), (384, 1), 0), reinterpret_tensor(arg144_1, (384, 1536), (1, 384), 0), out=buf95)
    del arg144_1
    buf96 = reinterpret_tensor(buf95, (8, 196, 1536), (301056, 1536, 1), 0); del buf95  # reuse
    cpp_fused_add_gelu_36(c_void_p(buf96.data_ptr()), c_void_p(arg145_1.data_ptr()))
    del arg145_1
    buf97 = reinterpret_tensor(buf94, (1568, 384), (384, 1), 0); del buf94  # reuse
    # Source Nodes: [x_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg147_1, reinterpret_tensor(buf96, (1568, 1536), (1536, 1), 0), reinterpret_tensor(arg146_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf97)
    del arg146_1
    del arg147_1
    del buf96
    buf98 = empty((8, 384), device='cpu', dtype=torch.float32)
    buf99 = buf98; del buf98  # reuse
    cpp_fused_add_addcmul_mean_mul_37(c_void_p(buf99.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf97.data_ptr()))
    del arg66_1
    del arg69_1
    del arg72_1
    del arg73_1
    del buf90
    del buf93
    del buf97
    buf100 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [mul_22, mul_23, x_100, x_102, x_103, x_105, x_92], Original ATen: [aten.add, aten.addcmul, aten.addmm, aten.mean, aten.mul]
    extern_kernels.addmm(arg149_1, buf99, reinterpret_tensor(arg148_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf100)
    del arg148_1
    del arg149_1
    return (buf100, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resmlp_12_224', benchmark_compiled_module)
