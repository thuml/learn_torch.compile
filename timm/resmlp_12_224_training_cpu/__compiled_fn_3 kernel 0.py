
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


cpp_fused_0 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (768L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
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
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            tmp9.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2)];
                        auto tmp2 = in_ptr2[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp6 = in_ptr4[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr6[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr6[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = at::vec::Vectorized<float>(tmp3);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp0);
                    auto tmp8 = tmp7 + tmp6;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp1 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                    out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp6;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_addcmul_clone_mul_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 + tmp8;
                    auto tmp10 = tmp4 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    tmp11.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151 = args
    args.clear()
    assert_size_stride(primals_1, (384, ), (1, ))
    assert_size_stride(primals_2, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_3, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_4, (384, ), (1, ))
    assert_size_stride(primals_5, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_6, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_7, (384, ), (1, ))
    assert_size_stride(primals_8, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_9, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_10, (384, ), (1, ))
    assert_size_stride(primals_11, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_12, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_13, (384, ), (1, ))
    assert_size_stride(primals_14, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_15, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_16, (384, ), (1, ))
    assert_size_stride(primals_17, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_18, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_19, (384, ), (1, ))
    assert_size_stride(primals_20, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_21, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_22, (384, ), (1, ))
    assert_size_stride(primals_23, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_24, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_25, (384, ), (1, ))
    assert_size_stride(primals_26, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_27, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_29, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_30, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_31, (384, ), (1, ))
    assert_size_stride(primals_32, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_33, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_35, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_36, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_37, (384, ), (1, ))
    assert_size_stride(primals_38, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_39, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_41, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_42, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_44, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_45, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_46, (384, ), (1, ))
    assert_size_stride(primals_47, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_48, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_50, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_51, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_52, (384, ), (1, ))
    assert_size_stride(primals_53, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_54, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_56, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_57, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_59, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_60, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_62, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_63, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_64, (384, ), (1, ))
    assert_size_stride(primals_65, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_66, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_69, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_71, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_72, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_73, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_74, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_75, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_76, (384, ), (1, ))
    assert_size_stride(primals_77, (196, 196), (196, 1))
    assert_size_stride(primals_78, (196, ), (1, ))
    assert_size_stride(primals_79, (1536, 384), (384, 1))
    assert_size_stride(primals_80, (1536, ), (1, ))
    assert_size_stride(primals_81, (384, 1536), (1536, 1))
    assert_size_stride(primals_82, (384, ), (1, ))
    assert_size_stride(primals_83, (196, 196), (196, 1))
    assert_size_stride(primals_84, (196, ), (1, ))
    assert_size_stride(primals_85, (1536, 384), (384, 1))
    assert_size_stride(primals_86, (1536, ), (1, ))
    assert_size_stride(primals_87, (384, 1536), (1536, 1))
    assert_size_stride(primals_88, (384, ), (1, ))
    assert_size_stride(primals_89, (196, 196), (196, 1))
    assert_size_stride(primals_90, (196, ), (1, ))
    assert_size_stride(primals_91, (1536, 384), (384, 1))
    assert_size_stride(primals_92, (1536, ), (1, ))
    assert_size_stride(primals_93, (384, 1536), (1536, 1))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_95, (196, 196), (196, 1))
    assert_size_stride(primals_96, (196, ), (1, ))
    assert_size_stride(primals_97, (1536, 384), (384, 1))
    assert_size_stride(primals_98, (1536, ), (1, ))
    assert_size_stride(primals_99, (384, 1536), (1536, 1))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_101, (196, 196), (196, 1))
    assert_size_stride(primals_102, (196, ), (1, ))
    assert_size_stride(primals_103, (1536, 384), (384, 1))
    assert_size_stride(primals_104, (1536, ), (1, ))
    assert_size_stride(primals_105, (384, 1536), (1536, 1))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (196, 196), (196, 1))
    assert_size_stride(primals_108, (196, ), (1, ))
    assert_size_stride(primals_109, (1536, 384), (384, 1))
    assert_size_stride(primals_110, (1536, ), (1, ))
    assert_size_stride(primals_111, (384, 1536), (1536, 1))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (196, 196), (196, 1))
    assert_size_stride(primals_114, (196, ), (1, ))
    assert_size_stride(primals_115, (1536, 384), (384, 1))
    assert_size_stride(primals_116, (1536, ), (1, ))
    assert_size_stride(primals_117, (384, 1536), (1536, 1))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (196, 196), (196, 1))
    assert_size_stride(primals_120, (196, ), (1, ))
    assert_size_stride(primals_121, (1536, 384), (384, 1))
    assert_size_stride(primals_122, (1536, ), (1, ))
    assert_size_stride(primals_123, (384, 1536), (1536, 1))
    assert_size_stride(primals_124, (384, ), (1, ))
    assert_size_stride(primals_125, (196, 196), (196, 1))
    assert_size_stride(primals_126, (196, ), (1, ))
    assert_size_stride(primals_127, (1536, 384), (384, 1))
    assert_size_stride(primals_128, (1536, ), (1, ))
    assert_size_stride(primals_129, (384, 1536), (1536, 1))
    assert_size_stride(primals_130, (384, ), (1, ))
    assert_size_stride(primals_131, (196, 196), (196, 1))
    assert_size_stride(primals_132, (196, ), (1, ))
    assert_size_stride(primals_133, (1536, 384), (384, 1))
    assert_size_stride(primals_134, (1536, ), (1, ))
    assert_size_stride(primals_135, (384, 1536), (1536, 1))
    assert_size_stride(primals_136, (384, ), (1, ))
    assert_size_stride(primals_137, (196, 196), (196, 1))
    assert_size_stride(primals_138, (196, ), (1, ))
    assert_size_stride(primals_139, (1536, 384), (384, 1))
    assert_size_stride(primals_140, (1536, ), (1, ))
    assert_size_stride(primals_141, (384, 1536), (1536, 1))
    assert_size_stride(primals_142, (384, ), (1, ))
    assert_size_stride(primals_143, (196, 196), (196, 1))
    assert_size_stride(primals_144, (196, ), (1, ))
    assert_size_stride(primals_145, (1536, 384), (384, 1))
    assert_size_stride(primals_146, (1536, ), (1, ))
    assert_size_stride(primals_147, (384, 1536), (1536, 1))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_149, (1000, 384), (384, 1))
    assert_size_stride(primals_150, (1000, ), (1, ))
    assert_size_stride(primals_151, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((384, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_75.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_151
    del primals_75
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, primals_76, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_76
    buf3 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_view_1(c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()))
    del primals_2
    buf4 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___0___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_78, buf3, reinterpret_tensor(primals_77, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf4)
    del primals_78
    buf5 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_2(c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del primals_5
    buf6 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf5, reinterpret_tensor(primals_79, (384, 1536), (1, 384), 0), out=buf6)
    buf7 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_3(c_void_p(buf6.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf7.data_ptr()))
    buf8 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_82, buf7, reinterpret_tensor(primals_81, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf8)
    del primals_82
    buf9 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf10 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_4(c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    del primals_8
    buf11 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___1___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_84, buf10, reinterpret_tensor(primals_83, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf11)
    del primals_84
    buf12 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_5(c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del primals_11
    buf13 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_13], Original ATen: [aten.mm]
    extern_kernels.mm(buf12, reinterpret_tensor(primals_85, (384, 1536), (1, 384), 0), out=buf13)
    buf14 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_6(c_void_p(buf13.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf14.data_ptr()))
    buf15 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_88, buf14, reinterpret_tensor(primals_87, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf15)
    del primals_88
    buf16 = buf9; del buf9  # reuse
    buf17 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_7(c_void_p(buf16.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf17.data_ptr()))
    del primals_14
    buf18 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___2___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_90, buf17, reinterpret_tensor(primals_89, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf18)
    del primals_90
    buf19 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_8(c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    del primals_17
    buf20 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_21], Original ATen: [aten.mm]
    extern_kernels.mm(buf19, reinterpret_tensor(primals_91, (384, 1536), (1, 384), 0), out=buf20)
    buf21 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_9(c_void_p(buf20.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf21.data_ptr()))
    buf22 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_94, buf21, reinterpret_tensor(primals_93, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf22)
    del primals_94
    buf23 = buf16; del buf16  # reuse
    buf24 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_10(c_void_p(buf23.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf24.data_ptr()))
    del primals_20
    buf25 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___3___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_96, buf24, reinterpret_tensor(primals_95, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf25)
    del primals_96
    buf26 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_11(c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del primals_23
    buf27 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_29], Original ATen: [aten.mm]
    extern_kernels.mm(buf26, reinterpret_tensor(primals_97, (384, 1536), (1, 384), 0), out=buf27)
    buf28 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_12(c_void_p(buf27.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_100, buf28, reinterpret_tensor(primals_99, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf29)
    del primals_100
    buf30 = buf23; del buf23  # reuse
    buf31 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_13(c_void_p(buf30.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf31.data_ptr()))
    del primals_26
    buf32 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___4___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, buf31, reinterpret_tensor(primals_101, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf32)
    del primals_102
    buf33 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_14(c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    del primals_29
    buf34 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_37], Original ATen: [aten.mm]
    extern_kernels.mm(buf33, reinterpret_tensor(primals_103, (384, 1536), (1, 384), 0), out=buf34)
    buf35 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_15(c_void_p(buf34.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf35.data_ptr()))
    buf36 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_106, buf35, reinterpret_tensor(primals_105, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf36)
    del primals_106
    buf37 = buf30; del buf30  # reuse
    buf38 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_16(c_void_p(buf37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf38.data_ptr()))
    del primals_32
    buf39 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___5___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_108, buf38, reinterpret_tensor(primals_107, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf39)
    del primals_108
    buf40 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_17(c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    del primals_35
    buf41 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_45], Original ATen: [aten.mm]
    extern_kernels.mm(buf40, reinterpret_tensor(primals_109, (384, 1536), (1, 384), 0), out=buf41)
    buf42 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_18(c_void_p(buf41.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf42.data_ptr()))
    buf43 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_49], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_112, buf42, reinterpret_tensor(primals_111, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf43)
    del primals_112
    buf44 = buf37; del buf37  # reuse
    buf45 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_19(c_void_p(buf44.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf45.data_ptr()))
    del primals_38
    buf46 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___6___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_114, buf45, reinterpret_tensor(primals_113, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf46)
    del primals_114
    buf47 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_20(c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    del primals_41
    buf48 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_53], Original ATen: [aten.mm]
    extern_kernels.mm(buf47, reinterpret_tensor(primals_115, (384, 1536), (1, 384), 0), out=buf48)
    buf49 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_21(c_void_p(buf48.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf49.data_ptr()))
    buf50 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_118, buf49, reinterpret_tensor(primals_117, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf50)
    del primals_118
    buf51 = buf44; del buf44  # reuse
    buf52 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_22(c_void_p(buf51.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf52.data_ptr()))
    del primals_44
    buf53 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___7___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf52, reinterpret_tensor(primals_119, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf53)
    del primals_120
    buf54 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_23(c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del primals_47
    buf55 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_61], Original ATen: [aten.mm]
    extern_kernels.mm(buf54, reinterpret_tensor(primals_121, (384, 1536), (1, 384), 0), out=buf55)
    buf56 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_24(c_void_p(buf55.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf56.data_ptr()))
    buf57 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_65], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_124, buf56, reinterpret_tensor(primals_123, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf57)
    del primals_124
    buf58 = buf51; del buf51  # reuse
    buf59 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_25(c_void_p(buf58.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf59.data_ptr()))
    del primals_50
    buf60 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___8___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_126, buf59, reinterpret_tensor(primals_125, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf60)
    del primals_126
    buf61 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_26(c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    del primals_53
    buf62 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_69], Original ATen: [aten.mm]
    extern_kernels.mm(buf61, reinterpret_tensor(primals_127, (384, 1536), (1, 384), 0), out=buf62)
    buf63 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_27(c_void_p(buf62.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf63.data_ptr()))
    buf64 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf63, reinterpret_tensor(primals_129, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf64)
    del primals_130
    buf65 = buf58; del buf58  # reuse
    buf66 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_28(c_void_p(buf65.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf66.data_ptr()))
    del primals_56
    buf67 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___9___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf66, reinterpret_tensor(primals_131, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf67)
    del primals_132
    buf68 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_29(c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del primals_59
    buf69 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_77], Original ATen: [aten.mm]
    extern_kernels.mm(buf68, reinterpret_tensor(primals_133, (384, 1536), (1, 384), 0), out=buf69)
    buf70 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_30(c_void_p(buf69.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_136, buf70, reinterpret_tensor(primals_135, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf71)
    del primals_136
    buf72 = buf65; del buf65  # reuse
    buf73 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_31(c_void_p(buf72.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_62
    buf74 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___10___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_138, buf73, reinterpret_tensor(primals_137, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf74)
    del primals_138
    buf75 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_32(c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del primals_65
    buf76 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_85], Original ATen: [aten.mm]
    extern_kernels.mm(buf75, reinterpret_tensor(primals_139, (384, 1536), (1, 384), 0), out=buf76)
    buf77 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_33(c_void_p(buf76.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf77.data_ptr()))
    buf78 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_142, buf77, reinterpret_tensor(primals_141, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf78)
    del primals_142
    buf79 = buf72; del buf72  # reuse
    buf80 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_34(c_void_p(buf79.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf80.data_ptr()))
    del primals_68
    buf81 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___11___linear_tokens], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf80, reinterpret_tensor(primals_143, (196, 196), (1, 196), 0), alpha=1, beta=1, out=buf81)
    del primals_144
    buf82 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_addcmul_clone_mul_35(c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()))
    del primals_71
    buf83 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_93], Original ATen: [aten.mm]
    extern_kernels.mm(buf82, reinterpret_tensor(primals_145, (384, 1536), (1, 384), 0), out=buf83)
    buf84 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_36(c_void_p(buf83.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf84.data_ptr()))
    buf85 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_148, buf84, reinterpret_tensor(primals_147, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf85)
    del primals_148
    buf86 = empty((8, 384), device='cpu', dtype=torch.float32)
    buf87 = buf86; del buf86  # reuse
    cpp_fused_add_addcmul_mean_mul_37(c_void_p(buf87.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf85.data_ptr()))
    del buf79
    del primals_73
    buf88 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_150, buf87, reinterpret_tensor(primals_149, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf88)
    del primals_150
    return (buf88, primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_74, buf0, primals_80, primals_86, primals_92, primals_98, primals_104, primals_110, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf10, buf11, buf12, buf13, buf14, buf15, buf17, buf18, buf19, buf20, buf21, buf22, buf24, buf25, buf26, buf27, buf28, buf29, buf31, buf32, buf33, buf34, buf35, buf36, buf38, buf39, buf40, buf41, buf42, buf43, buf45, buf46, buf47, buf48, buf49, buf50, buf52, buf53, buf54, buf55, buf56, buf57, buf59, buf60, buf61, buf62, buf63, buf64, buf66, buf67, buf68, buf69, buf70, buf71, buf73, buf74, buf75, buf76, buf77, buf78, buf80, buf81, buf82, buf83, buf84, buf85, buf87, reinterpret_tensor(primals_149, (1000, 384), (384, 1), 0), reinterpret_tensor(primals_147, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_145, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_143, (196, 196), (196, 1), 0), reinterpret_tensor(primals_141, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_139, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_137, (196, 196), (196, 1), 0), reinterpret_tensor(primals_135, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_133, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_131, (196, 196), (196, 1), 0), reinterpret_tensor(primals_129, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_127, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_125, (196, 196), (196, 1), 0), reinterpret_tensor(primals_123, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_121, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_119, (196, 196), (196, 1), 0), reinterpret_tensor(primals_117, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_115, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_113, (196, 196), (196, 1), 0), reinterpret_tensor(primals_111, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_109, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_107, (196, 196), (196, 1), 0), reinterpret_tensor(primals_105, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_103, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_101, (196, 196), (196, 1), 0), reinterpret_tensor(primals_99, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_97, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_95, (196, 196), (196, 1), 0), reinterpret_tensor(primals_93, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_91, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_89, (196, 196), (196, 1), 0), reinterpret_tensor(primals_87, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_85, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_83, (196, 196), (196, 1), 0), reinterpret_tensor(primals_81, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_79, (1536, 384), (384, 1), 0), reinterpret_tensor(primals_77, (196, 196), (196, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resmlp_12_224', benchmark_compiled_module)
