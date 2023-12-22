
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


cpp_fused_div_mul_0 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(196.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = tmp3 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp10.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_2 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner + (384L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp1 = static_cast<float>(196.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = static_cast<float>(1.0);
                            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                            auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp4);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp8 * tmp11;
                            auto tmp13 = at::vec::Vectorized<float>(tmp6);
                            auto tmp14 = tmp13 + tmp12;
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp1 = static_cast<float>(196.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = tmp3 * tmp7;
                        auto tmp11 = tmp10 * tmp6;
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = tmp8 + tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_3 = async_compile.cpp('''
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
                        float tmp14[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp14, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(tmp14 + static_cast<long>(8L*x1_inner));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp1 = static_cast<float>(196.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = static_cast<float>(1.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = tmp3 * tmp7;
                            auto tmp11 = tmp10 * tmp6;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp8 + tmp12;
                            auto tmp17 = tmp16 * tmp6;
                            auto tmp18 = tmp15 * tmp17;
                            auto tmp19 = tmp13 + tmp18;
                            tmp19.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp8 = in_ptr3[static_cast<long>(x2)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp13 = in_ptr5[static_cast<long>(x2)];
                        auto tmp1 = static_cast<float>(196.0);
                        auto tmp2 = tmp0 / tmp1;
                        auto tmp4 = static_cast<float>(1.0);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp4);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        auto tmp14 = decltype(tmp13)(tmp13 * tmp4);
                        auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                        auto tmp16 = decltype(tmp11)(tmp11 + tmp15);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp16;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_mul_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
                       const float* in_ptr33,
                       const float* in_ptr34,
                       const float* in_ptr35,
                       const float* in_ptr36,
                       const float* in_ptr37,
                       const float* in_ptr38,
                       const float* in_ptr39,
                       const float* in_ptr40,
                       const float* in_ptr41,
                       const float* in_ptr42,
                       const float* in_ptr43,
                       const float* in_ptr44,
                       const float* in_ptr45,
                       const float* in_ptr46,
                       const float* in_ptr47,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11)
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
                        float tmp11[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp11, 8);
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr10 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp20, 8);
                        float tmp29[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr14 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp29, 8);
                        float tmp38[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr18 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp38, 8);
                        float tmp47[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr22 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp47, 8);
                        float tmp56[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr26 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp56, 8);
                        float tmp65[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr30 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp65, 8);
                        float tmp74[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr34 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp74, 8);
                        float tmp83[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr38 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp83, 8);
                        float tmp92[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr42 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp92, 8);
                        float tmp109[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(tmp11 + static_cast<long>(8L*x1_inner));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x2));
                            auto tmp30 = at::vec::Vectorized<float>::loadu(tmp29 + static_cast<long>(8L*x1_inner));
                            auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x2));
                            auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x2));
                            auto tmp39 = at::vec::Vectorized<float>::loadu(tmp38 + static_cast<long>(8L*x1_inner));
                            auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x2));
                            auto tmp43 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp46 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x2));
                            auto tmp48 = at::vec::Vectorized<float>::loadu(tmp47 + static_cast<long>(8L*x1_inner));
                            auto tmp51 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x2));
                            auto tmp52 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp55 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x2));
                            auto tmp57 = at::vec::Vectorized<float>::loadu(tmp56 + static_cast<long>(8L*x1_inner));
                            auto tmp60 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x2));
                            auto tmp61 = at::vec::Vectorized<float>::loadu(in_ptr28 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp64 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x2));
                            auto tmp66 = at::vec::Vectorized<float>::loadu(tmp65 + static_cast<long>(8L*x1_inner));
                            auto tmp69 = at::vec::Vectorized<float>::loadu(in_ptr31 + static_cast<long>(x2));
                            auto tmp70 = at::vec::Vectorized<float>::loadu(in_ptr32 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp73 = at::vec::Vectorized<float>::loadu(in_ptr33 + static_cast<long>(x2));
                            auto tmp75 = at::vec::Vectorized<float>::loadu(tmp74 + static_cast<long>(8L*x1_inner));
                            auto tmp78 = at::vec::Vectorized<float>::loadu(in_ptr35 + static_cast<long>(x2));
                            auto tmp79 = at::vec::Vectorized<float>::loadu(in_ptr36 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp82 = at::vec::Vectorized<float>::loadu(in_ptr37 + static_cast<long>(x2));
                            auto tmp84 = at::vec::Vectorized<float>::loadu(tmp83 + static_cast<long>(8L*x1_inner));
                            auto tmp87 = at::vec::Vectorized<float>::loadu(in_ptr39 + static_cast<long>(x2));
                            auto tmp88 = at::vec::Vectorized<float>::loadu(in_ptr40 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp91 = at::vec::Vectorized<float>::loadu(in_ptr41 + static_cast<long>(x2));
                            auto tmp93 = at::vec::Vectorized<float>::loadu(tmp92 + static_cast<long>(8L*x1_inner));
                            auto tmp96 = at::vec::Vectorized<float>::loadu(in_ptr43 + static_cast<long>(x2));
                            auto tmp97 = at::vec::Vectorized<float>::loadu(in_ptr44 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp100 = at::vec::Vectorized<float>::loadu(in_ptr45 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp101 = at::vec::Vectorized<float>::loadu(in_ptr46 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp102 = at::vec::Vectorized<float>::loadu(in_ptr47 + static_cast<long>(x2));
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 + tmp4;
                            auto tmp8 = tmp6 * tmp7;
                            auto tmp9 = tmp5 + tmp8;
                            auto tmp13 = tmp10 * tmp12;
                            auto tmp14 = tmp9 + tmp13;
                            auto tmp17 = tmp15 * tmp16;
                            auto tmp18 = tmp14 + tmp17;
                            auto tmp22 = tmp19 * tmp21;
                            auto tmp23 = tmp18 + tmp22;
                            auto tmp26 = tmp24 * tmp25;
                            auto tmp27 = tmp23 + tmp26;
                            auto tmp31 = tmp28 * tmp30;
                            auto tmp32 = tmp27 + tmp31;
                            auto tmp35 = tmp33 * tmp34;
                            auto tmp36 = tmp32 + tmp35;
                            auto tmp40 = tmp37 * tmp39;
                            auto tmp41 = tmp36 + tmp40;
                            auto tmp44 = tmp42 * tmp43;
                            auto tmp45 = tmp41 + tmp44;
                            auto tmp49 = tmp46 * tmp48;
                            auto tmp50 = tmp45 + tmp49;
                            auto tmp53 = tmp51 * tmp52;
                            auto tmp54 = tmp50 + tmp53;
                            auto tmp58 = tmp55 * tmp57;
                            auto tmp59 = tmp54 + tmp58;
                            auto tmp62 = tmp60 * tmp61;
                            auto tmp63 = tmp59 + tmp62;
                            auto tmp67 = tmp64 * tmp66;
                            auto tmp68 = tmp63 + tmp67;
                            auto tmp71 = tmp69 * tmp70;
                            auto tmp72 = tmp68 + tmp71;
                            auto tmp76 = tmp73 * tmp75;
                            auto tmp77 = tmp72 + tmp76;
                            auto tmp80 = tmp78 * tmp79;
                            auto tmp81 = tmp77 + tmp80;
                            auto tmp85 = tmp82 * tmp84;
                            auto tmp86 = tmp81 + tmp85;
                            auto tmp89 = tmp87 * tmp88;
                            auto tmp90 = tmp86 + tmp89;
                            auto tmp94 = tmp91 * tmp93;
                            auto tmp95 = tmp90 + tmp94;
                            auto tmp98 = tmp96 * tmp97;
                            auto tmp99 = tmp95 + tmp98;
                            auto tmp103 = static_cast<float>(1.0);
                            auto tmp104 = at::vec::Vectorized<float>(tmp103);
                            auto tmp105 = tmp102 * tmp104;
                            auto tmp106 = tmp101 * tmp105;
                            auto tmp107 = tmp100 + tmp106;
                            auto tmp108 = tmp107 * tmp91;
                            tmp9.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp18.store(out_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp27.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp36.store(out_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp45.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp54.store(out_ptr5 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp63.store(out_ptr6 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp72.store(out_ptr7 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp81.store(out_ptr8 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp90.store(out_ptr9 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp99.store(out_ptr10 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            tmp108.store(tmp109 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp109, 8, out_ptr11 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
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
                        auto tmp9 = in_ptr5[static_cast<long>(x2)];
                        auto tmp10 = in_ptr6[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp13 = in_ptr7[static_cast<long>(x2)];
                        auto tmp14 = in_ptr8[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp17 = in_ptr9[static_cast<long>(x2)];
                        auto tmp18 = in_ptr10[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp21 = in_ptr11[static_cast<long>(x2)];
                        auto tmp22 = in_ptr12[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp25 = in_ptr13[static_cast<long>(x2)];
                        auto tmp26 = in_ptr14[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp29 = in_ptr15[static_cast<long>(x2)];
                        auto tmp30 = in_ptr16[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp33 = in_ptr17[static_cast<long>(x2)];
                        auto tmp34 = in_ptr18[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp37 = in_ptr19[static_cast<long>(x2)];
                        auto tmp38 = in_ptr20[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp41 = in_ptr21[static_cast<long>(x2)];
                        auto tmp42 = in_ptr22[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp45 = in_ptr23[static_cast<long>(x2)];
                        auto tmp46 = in_ptr24[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp49 = in_ptr25[static_cast<long>(x2)];
                        auto tmp50 = in_ptr26[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp53 = in_ptr27[static_cast<long>(x2)];
                        auto tmp54 = in_ptr28[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp57 = in_ptr29[static_cast<long>(x2)];
                        auto tmp58 = in_ptr30[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp61 = in_ptr31[static_cast<long>(x2)];
                        auto tmp62 = in_ptr32[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp65 = in_ptr33[static_cast<long>(x2)];
                        auto tmp66 = in_ptr34[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp69 = in_ptr35[static_cast<long>(x2)];
                        auto tmp70 = in_ptr36[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp73 = in_ptr37[static_cast<long>(x2)];
                        auto tmp74 = in_ptr38[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp77 = in_ptr39[static_cast<long>(x2)];
                        auto tmp78 = in_ptr40[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp81 = in_ptr41[static_cast<long>(x2)];
                        auto tmp82 = in_ptr42[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp85 = in_ptr43[static_cast<long>(x2)];
                        auto tmp86 = in_ptr44[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp89 = in_ptr45[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp90 = in_ptr46[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp91 = in_ptr47[static_cast<long>(x2)];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 + tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 + tmp11);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp12)(tmp12 + tmp15);
                        auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                        auto tmp20 = decltype(tmp16)(tmp16 + tmp19);
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                        auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                        auto tmp28 = decltype(tmp24)(tmp24 + tmp27);
                        auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                        auto tmp32 = decltype(tmp28)(tmp28 + tmp31);
                        auto tmp35 = decltype(tmp33)(tmp33 * tmp34);
                        auto tmp36 = decltype(tmp32)(tmp32 + tmp35);
                        auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                        auto tmp40 = decltype(tmp36)(tmp36 + tmp39);
                        auto tmp43 = decltype(tmp41)(tmp41 * tmp42);
                        auto tmp44 = decltype(tmp40)(tmp40 + tmp43);
                        auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                        auto tmp48 = decltype(tmp44)(tmp44 + tmp47);
                        auto tmp51 = decltype(tmp49)(tmp49 * tmp50);
                        auto tmp52 = decltype(tmp48)(tmp48 + tmp51);
                        auto tmp55 = decltype(tmp53)(tmp53 * tmp54);
                        auto tmp56 = decltype(tmp52)(tmp52 + tmp55);
                        auto tmp59 = decltype(tmp57)(tmp57 * tmp58);
                        auto tmp60 = decltype(tmp56)(tmp56 + tmp59);
                        auto tmp63 = decltype(tmp61)(tmp61 * tmp62);
                        auto tmp64 = decltype(tmp60)(tmp60 + tmp63);
                        auto tmp67 = decltype(tmp65)(tmp65 * tmp66);
                        auto tmp68 = decltype(tmp64)(tmp64 + tmp67);
                        auto tmp71 = decltype(tmp69)(tmp69 * tmp70);
                        auto tmp72 = decltype(tmp68)(tmp68 + tmp71);
                        auto tmp75 = decltype(tmp73)(tmp73 * tmp74);
                        auto tmp76 = decltype(tmp72)(tmp72 + tmp75);
                        auto tmp79 = decltype(tmp77)(tmp77 * tmp78);
                        auto tmp80 = decltype(tmp76)(tmp76 + tmp79);
                        auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                        auto tmp84 = decltype(tmp80)(tmp80 + tmp83);
                        auto tmp87 = decltype(tmp85)(tmp85 * tmp86);
                        auto tmp88 = decltype(tmp84)(tmp84 + tmp87);
                        auto tmp92 = static_cast<float>(1.0);
                        auto tmp93 = decltype(tmp91)(tmp91 * tmp92);
                        auto tmp94 = decltype(tmp90)(tmp90 * tmp93);
                        auto tmp95 = decltype(tmp89)(tmp89 + tmp94);
                        auto tmp96 = decltype(tmp95)(tmp95 * tmp81);
                        out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                        out_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp16;
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp24;
                        out_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp32;
                        out_ptr4[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp40;
                        out_ptr5[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp48;
                        out_ptr6[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp56;
                        out_ptr7[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp64;
                        out_ptr8[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp72;
                        out_ptr9[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp80;
                        out_ptr10[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp88;
                        out_ptr11[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp96;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_sum_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1000L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1000L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc4 = 0;
                    at::vec::Vectorized<float> tmp_acc4_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc5 = 0;
                    at::vec::Vectorized<float> tmp_acc5_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc6 = 0;
                    at::vec::Vectorized<float> tmp_acc6_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp6[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp6, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp6, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp6, 8);
                            float tmp30[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr10 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp30, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr10 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp30, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x2_inner));
                                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                                auto tmp31 = at::vec::Vectorized<float>::loadu(tmp30 + static_cast<long>(8L*x2_inner));
                                auto tmp1 = static_cast<float>(196.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp8 = tmp5 * tmp7;
                                auto tmp9 = tmp4 + tmp8;
                                auto tmp12 = tmp10 * tmp11;
                                auto tmp13 = tmp9 + tmp12;
                                auto tmp14 = static_cast<float>(1.0);
                                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                                auto tmp16 = tmp13 * tmp15;
                                auto tmp17 = tmp3 * tmp16;
                                auto tmp19 = tmp9 * tmp15;
                                auto tmp20 = tmp18 * tmp19;
                                auto tmp22 = tmp21 * tmp15;
                                auto tmp23 = tmp3 * tmp22;
                                auto tmp25 = tmp24 * tmp15;
                                auto tmp26 = tmp18 * tmp25;
                                auto tmp27 = tmp23 + tmp26;
                                auto tmp28 = tmp27 * tmp7;
                                auto tmp29 = tmp23 * tmp11;
                                auto tmp32 = tmp4 * tmp15;
                                auto tmp33 = tmp31 * tmp32;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp17;
                                tmp_acc2_vec = tmp_acc2_vec + tmp20;
                                tmp_acc3_vec = tmp_acc3_vec + tmp28;
                                tmp_acc4_vec = tmp_acc4_vec + tmp29;
                                tmp_acc5_vec = tmp_acc5_vec + tmp31;
                                tmp_acc6_vec = tmp_acc6_vec + tmp33;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr4[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                            auto tmp29 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr10[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(196.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp8 = tmp4 + tmp7;
                            auto tmp11 = tmp9 * tmp10;
                            auto tmp12 = tmp8 + tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp3 * tmp15;
                            auto tmp18 = tmp8 * tmp14;
                            auto tmp19 = tmp17 * tmp18;
                            auto tmp21 = tmp20 * tmp14;
                            auto tmp22 = tmp3 * tmp21;
                            auto tmp24 = tmp23 * tmp14;
                            auto tmp25 = tmp17 * tmp24;
                            auto tmp26 = tmp22 + tmp25;
                            auto tmp27 = tmp26 * tmp6;
                            auto tmp28 = tmp22 * tmp10;
                            auto tmp30 = tmp4 * tmp14;
                            auto tmp31 = tmp29 * tmp30;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp16;
                            tmp_acc2_vec = tmp_acc2_vec + tmp19;
                            tmp_acc3_vec = tmp_acc3_vec + tmp27;
                            tmp_acc4_vec = tmp_acc4_vec + tmp28;
                            tmp_acc5_vec = tmp_acc5_vec + tmp29;
                            tmp_acc6_vec = tmp_acc6_vec + tmp31;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc4_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc5_vec.store(out_ptr6 + static_cast<long>(x0));
                    tmp_acc6_vec.store(out_ptr7 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_sum_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp3[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp3, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr6 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp17, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x2_inner));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp6 = tmp1 + tmp5;
                                auto tmp7 = static_cast<float>(1.0);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp0 * tmp9;
                                auto tmp13 = tmp12 * tmp8;
                                auto tmp14 = tmp0 * tmp13;
                                auto tmp15 = tmp11 + tmp14;
                                auto tmp16 = tmp15 * tmp4;
                                auto tmp19 = tmp1 * tmp8;
                                auto tmp20 = tmp18 * tmp19;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                                tmp_acc2_vec = tmp_acc2_vec + tmp18;
                                tmp_acc3_vec = tmp_acc3_vec + tmp20;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr6[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp0 * tmp8;
                            auto tmp12 = tmp11 * tmp7;
                            auto tmp13 = tmp0 * tmp12;
                            auto tmp14 = tmp10 + tmp13;
                            auto tmp15 = tmp14 * tmp3;
                            auto tmp17 = tmp1 * tmp7;
                            auto tmp18 = tmp16 * tmp17;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp18;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_sum_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
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
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp8, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(1.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp11 = tmp10 * tmp4;
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x2)];
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp1)(tmp1 * tmp4);
                        auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp6)(tmp6 + tmp10);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_74, primals_75, primals_80, primals_86, primals_92, primals_98, primals_104, primals_110, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_151, convolution, view_1, addmm, view_3, mm, view_5, addmm_1, view_7, addmm_2, view_9, mm_1, view_11, addmm_3, view_13, addmm_4, view_15, mm_2, view_17, addmm_5, view_19, addmm_6, view_21, mm_3, view_23, addmm_7, view_25, addmm_8, view_27, mm_4, view_29, addmm_9, view_31, addmm_10, view_33, mm_5, view_35, addmm_11, view_37, addmm_12, view_39, mm_6, view_41, addmm_13, view_43, addmm_14, view_45, mm_7, view_47, addmm_15, view_49, addmm_16, view_51, mm_8, view_53, addmm_17, view_55, addmm_18, view_57, mm_9, view_59, addmm_19, view_61, addmm_20, view_63, mm_10, view_65, addmm_21, view_67, addmm_22, view_69, mm_11, view_71, addmm_23, clone_36, permute_62, permute_66, permute_72, permute_75, permute_80, permute_86, permute_89, permute_94, permute_100, permute_103, permute_108, permute_114, permute_117, permute_122, permute_128, permute_131, permute_136, permute_142, permute_145, permute_150, permute_156, permute_159, permute_164, permute_170, permute_173, permute_178, permute_184, permute_187, permute_192, permute_198, permute_201, permute_206, permute_212, permute_215, permute_220, permute_226, permute_229, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (384, ), (1, ))
    assert_size_stride(primals_3, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_4, (384, ), (1, ))
    assert_size_stride(primals_6, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_7, (384, ), (1, ))
    assert_size_stride(primals_9, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_10, (384, ), (1, ))
    assert_size_stride(primals_12, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_13, (384, ), (1, ))
    assert_size_stride(primals_15, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_16, (384, ), (1, ))
    assert_size_stride(primals_18, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_19, (384, ), (1, ))
    assert_size_stride(primals_21, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_22, (384, ), (1, ))
    assert_size_stride(primals_24, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_25, (384, ), (1, ))
    assert_size_stride(primals_27, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_30, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_31, (384, ), (1, ))
    assert_size_stride(primals_33, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_36, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_37, (384, ), (1, ))
    assert_size_stride(primals_39, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_42, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_45, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_46, (384, ), (1, ))
    assert_size_stride(primals_48, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_51, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_52, (384, ), (1, ))
    assert_size_stride(primals_54, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_57, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_60, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_63, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_64, (384, ), (1, ))
    assert_size_stride(primals_66, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_69, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_72, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_74, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_75, (384, 3, 16, 16), (768, 1, 48, 3))
    assert_size_stride(primals_80, (1536, ), (1, ))
    assert_size_stride(primals_86, (1536, ), (1, ))
    assert_size_stride(primals_92, (1536, ), (1, ))
    assert_size_stride(primals_98, (1536, ), (1, ))
    assert_size_stride(primals_104, (1536, ), (1, ))
    assert_size_stride(primals_110, (1536, ), (1, ))
    assert_size_stride(primals_116, (1536, ), (1, ))
    assert_size_stride(primals_122, (1536, ), (1, ))
    assert_size_stride(primals_128, (1536, ), (1, ))
    assert_size_stride(primals_134, (1536, ), (1, ))
    assert_size_stride(primals_140, (1536, ), (1, ))
    assert_size_stride(primals_146, (1536, ), (1, ))
    assert_size_stride(primals_151, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(view_1, (3072, 196), (196, 1))
    assert_size_stride(addmm, (3072, 196), (196, 1))
    assert_size_stride(view_3, (1568, 384), (384, 1))
    assert_size_stride(mm, (1568, 1536), (1536, 1))
    assert_size_stride(view_5, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_1, (1568, 384), (384, 1))
    assert_size_stride(view_7, (3072, 196), (196, 1))
    assert_size_stride(addmm_2, (3072, 196), (196, 1))
    assert_size_stride(view_9, (1568, 384), (384, 1))
    assert_size_stride(mm_1, (1568, 1536), (1536, 1))
    assert_size_stride(view_11, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_3, (1568, 384), (384, 1))
    assert_size_stride(view_13, (3072, 196), (196, 1))
    assert_size_stride(addmm_4, (3072, 196), (196, 1))
    assert_size_stride(view_15, (1568, 384), (384, 1))
    assert_size_stride(mm_2, (1568, 1536), (1536, 1))
    assert_size_stride(view_17, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_5, (1568, 384), (384, 1))
    assert_size_stride(view_19, (3072, 196), (196, 1))
    assert_size_stride(addmm_6, (3072, 196), (196, 1))
    assert_size_stride(view_21, (1568, 384), (384, 1))
    assert_size_stride(mm_3, (1568, 1536), (1536, 1))
    assert_size_stride(view_23, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_7, (1568, 384), (384, 1))
    assert_size_stride(view_25, (3072, 196), (196, 1))
    assert_size_stride(addmm_8, (3072, 196), (196, 1))
    assert_size_stride(view_27, (1568, 384), (384, 1))
    assert_size_stride(mm_4, (1568, 1536), (1536, 1))
    assert_size_stride(view_29, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_9, (1568, 384), (384, 1))
    assert_size_stride(view_31, (3072, 196), (196, 1))
    assert_size_stride(addmm_10, (3072, 196), (196, 1))
    assert_size_stride(view_33, (1568, 384), (384, 1))
    assert_size_stride(mm_5, (1568, 1536), (1536, 1))
    assert_size_stride(view_35, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_11, (1568, 384), (384, 1))
    assert_size_stride(view_37, (3072, 196), (196, 1))
    assert_size_stride(addmm_12, (3072, 196), (196, 1))
    assert_size_stride(view_39, (1568, 384), (384, 1))
    assert_size_stride(mm_6, (1568, 1536), (1536, 1))
    assert_size_stride(view_41, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_13, (1568, 384), (384, 1))
    assert_size_stride(view_43, (3072, 196), (196, 1))
    assert_size_stride(addmm_14, (3072, 196), (196, 1))
    assert_size_stride(view_45, (1568, 384), (384, 1))
    assert_size_stride(mm_7, (1568, 1536), (1536, 1))
    assert_size_stride(view_47, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_15, (1568, 384), (384, 1))
    assert_size_stride(view_49, (3072, 196), (196, 1))
    assert_size_stride(addmm_16, (3072, 196), (196, 1))
    assert_size_stride(view_51, (1568, 384), (384, 1))
    assert_size_stride(mm_8, (1568, 1536), (1536, 1))
    assert_size_stride(view_53, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_17, (1568, 384), (384, 1))
    assert_size_stride(view_55, (3072, 196), (196, 1))
    assert_size_stride(addmm_18, (3072, 196), (196, 1))
    assert_size_stride(view_57, (1568, 384), (384, 1))
    assert_size_stride(mm_9, (1568, 1536), (1536, 1))
    assert_size_stride(view_59, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_19, (1568, 384), (384, 1))
    assert_size_stride(view_61, (3072, 196), (196, 1))
    assert_size_stride(addmm_20, (3072, 196), (196, 1))
    assert_size_stride(view_63, (1568, 384), (384, 1))
    assert_size_stride(mm_10, (1568, 1536), (1536, 1))
    assert_size_stride(view_65, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_21, (1568, 384), (384, 1))
    assert_size_stride(view_67, (3072, 196), (196, 1))
    assert_size_stride(addmm_22, (3072, 196), (196, 1))
    assert_size_stride(view_69, (1568, 384), (384, 1))
    assert_size_stride(mm_11, (1568, 1536), (1536, 1))
    assert_size_stride(view_71, (1568, 1536), (1536, 1))
    assert_size_stride(addmm_23, (1568, 384), (384, 1))
    assert_size_stride(clone_36, (8, 384), (384, 1))
    assert_size_stride(permute_62, (1000, 384), (384, 1))
    assert_size_stride(permute_66, (384, 1536), (1536, 1))
    assert_size_stride(permute_72, (1536, 384), (384, 1))
    assert_size_stride(permute_75, (196, 196), (196, 1))
    assert_size_stride(permute_80, (384, 1536), (1536, 1))
    assert_size_stride(permute_86, (1536, 384), (384, 1))
    assert_size_stride(permute_89, (196, 196), (196, 1))
    assert_size_stride(permute_94, (384, 1536), (1536, 1))
    assert_size_stride(permute_100, (1536, 384), (384, 1))
    assert_size_stride(permute_103, (196, 196), (196, 1))
    assert_size_stride(permute_108, (384, 1536), (1536, 1))
    assert_size_stride(permute_114, (1536, 384), (384, 1))
    assert_size_stride(permute_117, (196, 196), (196, 1))
    assert_size_stride(permute_122, (384, 1536), (1536, 1))
    assert_size_stride(permute_128, (1536, 384), (384, 1))
    assert_size_stride(permute_131, (196, 196), (196, 1))
    assert_size_stride(permute_136, (384, 1536), (1536, 1))
    assert_size_stride(permute_142, (1536, 384), (384, 1))
    assert_size_stride(permute_145, (196, 196), (196, 1))
    assert_size_stride(permute_150, (384, 1536), (1536, 1))
    assert_size_stride(permute_156, (1536, 384), (384, 1))
    assert_size_stride(permute_159, (196, 196), (196, 1))
    assert_size_stride(permute_164, (384, 1536), (1536, 1))
    assert_size_stride(permute_170, (1536, 384), (384, 1))
    assert_size_stride(permute_173, (196, 196), (196, 1))
    assert_size_stride(permute_178, (384, 1536), (1536, 1))
    assert_size_stride(permute_184, (1536, 384), (384, 1))
    assert_size_stride(permute_187, (196, 196), (196, 1))
    assert_size_stride(permute_192, (384, 1536), (1536, 1))
    assert_size_stride(permute_198, (1536, 384), (384, 1))
    assert_size_stride(permute_201, (196, 196), (196, 1))
    assert_size_stride(permute_206, (384, 1536), (1536, 1))
    assert_size_stride(permute_212, (1536, 384), (384, 1))
    assert_size_stride(permute_215, (196, 196), (196, 1))
    assert_size_stride(permute_220, (384, 1536), (1536, 1))
    assert_size_stride(permute_226, (1536, 384), (384, 1))
    assert_size_stride(permute_229, (196, 196), (196, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf11 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_62, out=buf11)
    del permute_62
    buf17 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    cpp_fused_div_mul_0(c_void_p(buf11.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf17.data_ptr()))
    buf18 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf17, (1568, 384), (384, 1), 0), permute_66, out=buf18)
    del permute_66
    buf21 = reinterpret_tensor(buf18, (8, 196, 1536), (301056, 1536, 1), 0); del buf18  # reuse
    cpp_fused_add_gelu_gelu_backward_1(c_void_p(buf21.data_ptr()), c_void_p(mm_11.data_ptr()), c_void_p(primals_146.data_ptr()))
    del mm_11
    del primals_146
    buf24 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (1568, 1536), (1536, 1), 0), permute_72, out=buf24)
    del permute_72
    buf28 = empty((8, 384, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_2(c_void_p(buf11.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf28, (3072, 196), (196, 1), 0), permute_75, out=buf29)
    del permute_75
    buf34 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf36 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_3(c_void_p(buf11.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()))
    del primals_69
    buf37 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (1568, 384), (384, 1), 0), permute_80, out=buf37)
    del permute_80
    buf40 = reinterpret_tensor(buf37, (8, 196, 1536), (301056, 1536, 1), 0); del buf37  # reuse
    cpp_fused_add_gelu_gelu_backward_4(c_void_p(buf40.data_ptr()), c_void_p(mm_10.data_ptr()), c_void_p(primals_140.data_ptr()))
    del mm_10
    del primals_140
    buf43 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf40, (1568, 1536), (1536, 1), 0), permute_86, out=buf43)
    del permute_86
    buf0 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf1 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf2 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf3 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf4 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf5 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf7 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf8 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf9 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf10 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf47 = empty((8, 384, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_mul_5(c_void_p(convolution.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(addmm.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(addmm_3.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(addmm_5.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(addmm_8.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(addmm_9.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(addmm_11.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(addmm_12.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(addmm_15.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(addmm_17.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(addmm_20.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(addmm_21.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf47.data_ptr()))
    del primals_64
    buf12 = empty((1000, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_36, out=buf12)
    del clone_36
    buf13 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf14 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf15 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf26 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf27 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf16 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf32 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf33 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_sum_6(c_void_p(tangents_1.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(addmm_23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    del addmm_22
    del addmm_23
    del buf10
    del buf11
    del buf29
    del primals_67
    del primals_70
    del primals_72
    del primals_74
    del tangents_1
    buf19 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf17, (384, 1568), (1, 384), 0), view_71, out=buf19)
    del view_71
    buf20 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf22 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_sum_7(c_void_p(buf17.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()))
    del buf17
    buf23 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (1536, 1568), (1, 1536), 0), view_69, out=buf23)
    del buf21
    del view_69
    buf25 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_8(c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del buf24
    buf30 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf28, (196, 3072), (1, 196), 0), view_67, out=buf30)
    del view_67
    buf31 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf35 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_mul_sum_9(c_void_p(buf28.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(addmm_21.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf35.data_ptr()))
    del addmm_21
    del buf28
    buf38 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (384, 1568), (1, 384), 0), view_65, out=buf38)
    del view_65
    buf39 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf41 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_sum_10(c_void_p(buf36.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf41.data_ptr()))
    buf42 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf40, (1536, 1568), (1, 1536), 0), view_63, out=buf42)
    del view_63
    buf44 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_11(c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    buf48 = reinterpret_tensor(buf36, (3072, 196), (196, 1), 0); del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (3072, 196), (196, 1), 0), permute_89, out=buf48)
    del permute_89
    buf45 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf46 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf51 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf52 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_12(c_void_p(buf43.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(addmm_20.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    del addmm_20
    del primals_61
    buf49 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (196, 3072), (1, 196), 0), view_61, out=buf49)
    del view_61
    buf50 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf53 = buf34; del buf34  # reuse
    buf54 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf55 = buf9; del buf9  # reuse
    cpp_fused_add_mul_sum_13(c_void_p(buf53.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del addmm_19
    del buf43
    del primals_58
    del primals_63
    del primals_66
    buf56 = reinterpret_tensor(buf40, (1568, 1536), (1536, 1), 0); del buf40  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (1568, 384), (384, 1), 0), permute_94, out=buf56)
    del permute_94
    buf57 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (384, 1568), (1, 384), 0), view_59, out=buf57)
    del view_59
    buf58 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf59 = reinterpret_tensor(buf56, (8, 196, 1536), (301056, 1536, 1), 0); del buf56  # reuse
    buf60 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_14(c_void_p(buf59.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(mm_9.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()))
    del mm_9
    del primals_134
    buf61 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf59, (1536, 1568), (1, 1536), 0), view_57, out=buf61)
    del view_57
    buf62 = reinterpret_tensor(buf55, (1568, 384), (384, 1), 0); del buf55  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf59, (1568, 1536), (1536, 1), 0), permute_100, out=buf62)
    del permute_100
    buf63 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf66 = reinterpret_tensor(buf48, (8, 384, 196), (75264, 196, 1), 0); del buf48  # reuse
    cpp_fused_clone_sum_15(c_void_p(buf62.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf66.data_ptr()))
    buf67 = reinterpret_tensor(buf47, (3072, 196), (196, 1), 0); del buf47  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (3072, 196), (196, 1), 0), permute_103, out=buf67)
    del permute_103
    buf64 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf65 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf70 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf71 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_16(c_void_p(buf62.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()))
    del addmm_18
    del primals_55
    buf68 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (196, 3072), (1, 196), 0), view_55, out=buf68)
    del view_55
    buf69 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf72 = buf53; del buf53  # reuse
    buf73 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf74 = buf8; del buf8  # reuse
    cpp_fused_add_mul_sum_17(c_void_p(buf72.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(addmm_17.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    del addmm_17
    del buf62
    del primals_52
    del primals_57
    del primals_60
    buf75 = reinterpret_tensor(buf59, (1568, 1536), (1536, 1), 0); del buf59  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (1568, 384), (384, 1), 0), permute_108, out=buf75)
    del permute_108
    buf76 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (384, 1568), (1, 384), 0), view_53, out=buf76)
    del view_53
    buf77 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf78 = reinterpret_tensor(buf75, (8, 196, 1536), (301056, 1536, 1), 0); del buf75  # reuse
    buf79 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_18(c_void_p(buf78.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(mm_8.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()))
    del mm_8
    del primals_128
    buf80 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf78, (1536, 1568), (1, 1536), 0), view_51, out=buf80)
    del view_51
    buf81 = reinterpret_tensor(buf74, (1568, 384), (384, 1), 0); del buf74  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf78, (1568, 1536), (1536, 1), 0), permute_114, out=buf81)
    del permute_114
    buf82 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf85 = reinterpret_tensor(buf67, (8, 384, 196), (75264, 196, 1), 0); del buf67  # reuse
    cpp_fused_clone_sum_19(c_void_p(buf81.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf85.data_ptr()))
    buf86 = reinterpret_tensor(buf66, (3072, 196), (196, 1), 0); del buf66  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf85, (3072, 196), (196, 1), 0), permute_117, out=buf86)
    del permute_117
    buf83 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf84 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf89 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf90 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_20(c_void_p(buf81.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del addmm_16
    del primals_49
    buf87 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf85, (196, 3072), (1, 196), 0), view_49, out=buf87)
    del view_49
    buf88 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf91 = buf72; del buf72  # reuse
    buf92 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf93 = buf7; del buf7  # reuse
    cpp_fused_add_mul_sum_21(c_void_p(buf91.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(addmm_15.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    del addmm_15
    del buf81
    del primals_46
    del primals_51
    del primals_54
    buf94 = reinterpret_tensor(buf78, (1568, 1536), (1536, 1), 0); del buf78  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf93, (1568, 384), (384, 1), 0), permute_122, out=buf94)
    del permute_122
    buf95 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf93, (384, 1568), (1, 384), 0), view_47, out=buf95)
    del view_47
    buf96 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf97 = reinterpret_tensor(buf94, (8, 196, 1536), (301056, 1536, 1), 0); del buf94  # reuse
    buf98 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_22(c_void_p(buf97.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(mm_7.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf98.data_ptr()))
    del mm_7
    del primals_122
    buf99 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (1536, 1568), (1, 1536), 0), view_45, out=buf99)
    del view_45
    buf100 = reinterpret_tensor(buf93, (1568, 384), (384, 1), 0); del buf93  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (1568, 1536), (1536, 1), 0), permute_128, out=buf100)
    del permute_128
    buf101 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf104 = reinterpret_tensor(buf86, (8, 384, 196), (75264, 196, 1), 0); del buf86  # reuse
    cpp_fused_clone_sum_23(c_void_p(buf100.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf104.data_ptr()))
    buf105 = reinterpret_tensor(buf85, (3072, 196), (196, 1), 0); del buf85  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (3072, 196), (196, 1), 0), permute_131, out=buf105)
    del permute_131
    buf102 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf103 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf108 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf109 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_24(c_void_p(buf100.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del addmm_14
    del primals_43
    buf106 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (196, 3072), (1, 196), 0), view_43, out=buf106)
    del view_43
    buf107 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf110 = reinterpret_tensor(buf100, (8, 196, 384), (75264, 384, 1), 0); del buf100  # reuse
    buf111 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf112 = buf6; del buf6  # reuse
    cpp_fused_add_mul_sum_25(c_void_p(buf110.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    del addmm_13
    del buf104
    del primals_40
    del primals_45
    del primals_48
    buf113 = reinterpret_tensor(buf97, (1568, 1536), (1536, 1), 0); del buf97  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (1568, 384), (384, 1), 0), permute_136, out=buf113)
    del permute_136
    buf114 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (384, 1568), (1, 384), 0), view_41, out=buf114)
    del view_41
    buf115 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf116 = reinterpret_tensor(buf113, (8, 196, 1536), (301056, 1536, 1), 0); del buf113  # reuse
    buf117 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_26(c_void_p(buf116.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(mm_6.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()))
    del mm_6
    del primals_116
    buf118 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf116, (1536, 1568), (1, 1536), 0), view_39, out=buf118)
    del view_39
    buf119 = reinterpret_tensor(buf112, (1568, 384), (384, 1), 0); del buf112  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf116, (1568, 1536), (1536, 1), 0), permute_142, out=buf119)
    del permute_142
    buf120 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf123 = reinterpret_tensor(buf91, (8, 384, 196), (75264, 196, 1), 0); del buf91  # reuse
    cpp_fused_clone_sum_27(c_void_p(buf119.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf123.data_ptr()))
    buf124 = buf105; del buf105  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (3072, 196), (196, 1), 0), permute_145, out=buf124)
    del permute_145
    buf121 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf122 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf127 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf128 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_28(c_void_p(buf119.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(addmm_12.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    del addmm_12
    del primals_37
    buf125 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (196, 3072), (1, 196), 0), view_37, out=buf125)
    del view_37
    buf126 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf129 = buf110; del buf110  # reuse
    buf130 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf131 = buf5; del buf5  # reuse
    cpp_fused_add_mul_sum_29(c_void_p(buf129.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(addmm_11.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del addmm_11
    del buf119
    del primals_34
    del primals_39
    del primals_42
    buf132 = reinterpret_tensor(buf116, (1568, 1536), (1536, 1), 0); del buf116  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (1568, 384), (384, 1), 0), permute_150, out=buf132)
    del permute_150
    buf133 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (384, 1568), (1, 384), 0), view_35, out=buf133)
    del view_35
    buf134 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf135 = reinterpret_tensor(buf132, (8, 196, 1536), (301056, 1536, 1), 0); del buf132  # reuse
    buf136 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_30(c_void_p(buf135.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(mm_5.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()))
    del mm_5
    del primals_110
    buf137 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (1536, 1568), (1, 1536), 0), view_33, out=buf137)
    del view_33
    buf138 = reinterpret_tensor(buf131, (1568, 384), (384, 1), 0); del buf131  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (1568, 1536), (1536, 1), 0), permute_156, out=buf138)
    del permute_156
    buf139 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf142 = reinterpret_tensor(buf124, (8, 384, 196), (75264, 196, 1), 0); del buf124  # reuse
    cpp_fused_clone_sum_31(c_void_p(buf138.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf142.data_ptr()))
    buf143 = reinterpret_tensor(buf123, (3072, 196), (196, 1), 0); del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (3072, 196), (196, 1), 0), permute_159, out=buf143)
    del permute_159
    buf140 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf141 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf146 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf147 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_32(c_void_p(buf138.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    del addmm_10
    del primals_31
    buf144 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (196, 3072), (1, 196), 0), view_31, out=buf144)
    del view_31
    buf145 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf148 = buf129; del buf129  # reuse
    buf149 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf150 = buf4; del buf4  # reuse
    cpp_fused_add_mul_sum_33(c_void_p(buf148.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(addmm_9.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    del addmm_9
    del buf138
    del primals_28
    del primals_33
    del primals_36
    buf151 = reinterpret_tensor(buf135, (1568, 1536), (1536, 1), 0); del buf135  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (1568, 384), (384, 1), 0), permute_164, out=buf151)
    del permute_164
    buf152 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (384, 1568), (1, 384), 0), view_29, out=buf152)
    del view_29
    buf153 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf154 = reinterpret_tensor(buf151, (8, 196, 1536), (301056, 1536, 1), 0); del buf151  # reuse
    buf155 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_34(c_void_p(buf154.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(mm_4.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf155.data_ptr()))
    del mm_4
    del primals_104
    buf156 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (1536, 1568), (1, 1536), 0), view_27, out=buf156)
    del view_27
    buf157 = reinterpret_tensor(buf150, (1568, 384), (384, 1), 0); del buf150  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (1568, 1536), (1536, 1), 0), permute_170, out=buf157)
    del permute_170
    buf158 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf161 = reinterpret_tensor(buf143, (8, 384, 196), (75264, 196, 1), 0); del buf143  # reuse
    cpp_fused_clone_sum_35(c_void_p(buf157.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf161.data_ptr()))
    buf162 = reinterpret_tensor(buf142, (3072, 196), (196, 1), 0); del buf142  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (3072, 196), (196, 1), 0), permute_173, out=buf162)
    del permute_173
    buf159 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf160 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf165 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf166 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_36(c_void_p(buf157.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(addmm_8.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    del addmm_8
    del primals_25
    buf163 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (196, 3072), (1, 196), 0), view_25, out=buf163)
    del view_25
    buf164 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf167 = buf148; del buf148  # reuse
    buf168 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf169 = buf3; del buf3  # reuse
    cpp_fused_add_mul_sum_37(c_void_p(buf167.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    del addmm_7
    del buf157
    del primals_22
    del primals_27
    del primals_30
    buf170 = reinterpret_tensor(buf154, (1568, 1536), (1536, 1), 0); del buf154  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf169, (1568, 384), (384, 1), 0), permute_178, out=buf170)
    del permute_178
    buf171 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf169, (384, 1568), (1, 384), 0), view_23, out=buf171)
    del view_23
    buf172 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf173 = reinterpret_tensor(buf170, (8, 196, 1536), (301056, 1536, 1), 0); del buf170  # reuse
    buf174 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_38(c_void_p(buf173.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(mm_3.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()))
    del mm_3
    del primals_98
    buf175 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (1536, 1568), (1, 1536), 0), view_21, out=buf175)
    del view_21
    buf176 = reinterpret_tensor(buf169, (1568, 384), (384, 1), 0); del buf169  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (1568, 1536), (1536, 1), 0), permute_184, out=buf176)
    del permute_184
    buf177 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf180 = reinterpret_tensor(buf162, (8, 384, 196), (75264, 196, 1), 0); del buf162  # reuse
    cpp_fused_clone_sum_39(c_void_p(buf176.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf180.data_ptr()))
    buf181 = reinterpret_tensor(buf161, (3072, 196), (196, 1), 0); del buf161  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf180, (3072, 196), (196, 1), 0), permute_187, out=buf181)
    del permute_187
    buf178 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf179 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf184 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf185 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_40(c_void_p(buf176.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    del addmm_6
    del primals_19
    buf182 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf180, (196, 3072), (1, 196), 0), view_19, out=buf182)
    del view_19
    buf183 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf186 = buf167; del buf167  # reuse
    buf187 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf188 = buf2; del buf2  # reuse
    cpp_fused_add_mul_sum_41(c_void_p(buf186.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(addmm_5.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    del addmm_5
    del buf176
    del primals_16
    del primals_21
    del primals_24
    buf189 = reinterpret_tensor(buf173, (1568, 1536), (1536, 1), 0); del buf173  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (1568, 384), (384, 1), 0), permute_192, out=buf189)
    del permute_192
    buf190 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (384, 1568), (1, 384), 0), view_17, out=buf190)
    del view_17
    buf191 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf192 = reinterpret_tensor(buf189, (8, 196, 1536), (301056, 1536, 1), 0); del buf189  # reuse
    buf193 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_42(c_void_p(buf192.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(mm_2.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf193.data_ptr()))
    del mm_2
    del primals_92
    buf194 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf192, (1536, 1568), (1, 1536), 0), view_15, out=buf194)
    del view_15
    buf195 = reinterpret_tensor(buf188, (1568, 384), (384, 1), 0); del buf188  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf192, (1568, 1536), (1536, 1), 0), permute_198, out=buf195)
    del permute_198
    buf196 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf199 = reinterpret_tensor(buf181, (8, 384, 196), (75264, 196, 1), 0); del buf181  # reuse
    cpp_fused_clone_sum_43(c_void_p(buf195.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf199.data_ptr()))
    buf200 = reinterpret_tensor(buf180, (3072, 196), (196, 1), 0); del buf180  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (3072, 196), (196, 1), 0), permute_201, out=buf200)
    del permute_201
    buf197 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf198 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf203 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf204 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_44(c_void_p(buf195.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    del addmm_4
    del primals_13
    buf201 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (196, 3072), (1, 196), 0), view_13, out=buf201)
    del view_13
    buf202 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf205 = buf186; del buf186  # reuse
    buf206 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf207 = buf1; del buf1  # reuse
    cpp_fused_add_mul_sum_45(c_void_p(buf205.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(addmm_3.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    del addmm_3
    del buf195
    del primals_10
    del primals_15
    del primals_18
    buf208 = reinterpret_tensor(buf192, (1568, 1536), (1536, 1), 0); del buf192  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (1568, 384), (384, 1), 0), permute_206, out=buf208)
    del permute_206
    buf209 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (384, 1568), (1, 384), 0), view_11, out=buf209)
    del view_11
    buf210 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf211 = reinterpret_tensor(buf208, (8, 196, 1536), (301056, 1536, 1), 0); del buf208  # reuse
    buf212 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_46(c_void_p(buf211.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(mm_1.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()))
    del mm_1
    del primals_86
    buf213 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (1536, 1568), (1, 1536), 0), view_9, out=buf213)
    del view_9
    buf214 = reinterpret_tensor(buf207, (1568, 384), (384, 1), 0); del buf207  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (1568, 1536), (1536, 1), 0), permute_212, out=buf214)
    del permute_212
    buf215 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf218 = reinterpret_tensor(buf200, (8, 384, 196), (75264, 196, 1), 0); del buf200  # reuse
    cpp_fused_clone_sum_47(c_void_p(buf214.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf218.data_ptr()))
    buf219 = reinterpret_tensor(buf199, (3072, 196), (196, 1), 0); del buf199  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (3072, 196), (196, 1), 0), permute_215, out=buf219)
    del permute_215
    buf216 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf217 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf222 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf223 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_48(c_void_p(buf214.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    del addmm_2
    del primals_7
    buf220 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (196, 3072), (1, 196), 0), view_7, out=buf220)
    del view_7
    buf221 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf224 = buf205; del buf205  # reuse
    buf225 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf226 = buf0; del buf0  # reuse
    cpp_fused_add_mul_sum_49(c_void_p(buf224.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    del addmm_1
    del buf214
    del primals_12
    del primals_4
    del primals_9
    buf227 = reinterpret_tensor(buf211, (1568, 1536), (1536, 1), 0); del buf211  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (1568, 384), (384, 1), 0), permute_220, out=buf227)
    del permute_220
    buf228 = empty((384, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (384, 1568), (1, 384), 0), view_5, out=buf228)
    del view_5
    buf229 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf230 = reinterpret_tensor(buf227, (8, 196, 1536), (301056, 1536, 1), 0); del buf227  # reuse
    buf231 = empty((1, 1, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_50(c_void_p(buf230.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(mm.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()))
    del mm
    del primals_80
    buf232 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (1536, 1568), (1, 1536), 0), view_3, out=buf232)
    del view_3
    buf233 = reinterpret_tensor(buf226, (1568, 384), (384, 1), 0); del buf226  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (1568, 1536), (1536, 1), 0), permute_226, out=buf233)
    del buf230
    del permute_226
    buf234 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf237 = reinterpret_tensor(buf219, (8, 384, 196), (75264, 196, 1), 0); del buf219  # reuse
    cpp_fused_clone_sum_51(c_void_p(buf233.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf237.data_ptr()))
    buf238 = reinterpret_tensor(buf218, (3072, 196), (196, 1), 0); del buf218  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (3072, 196), (196, 1), 0), permute_229, out=buf238)
    del permute_229
    buf235 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf236 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf241 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    buf242 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_52(c_void_p(buf233.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(addmm.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    del addmm
    del convolution
    del primals_1
    buf239 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (196, 3072), (1, 196), 0), view_1, out=buf239)
    del view_1
    buf240 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf243 = reinterpret_tensor(buf224, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf224  # reuse
    cpp_fused_convolution_backward_sum_53(c_void_p(buf243.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf240.data_ptr()))
    del buf233
    del buf237
    del buf238
    del primals_3
    del primals_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf244 = aten.convolution_backward(buf243, primals_151, primals_75, [384], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf243
    del primals_151
    del primals_75
    buf245 = buf244[1]
    buf246 = buf244[2]
    return (reinterpret_tensor(buf236, (384, ), (1, ), 0), buf241, buf242, reinterpret_tensor(buf225, (384, ), (1, ), 0), buf234, buf235, reinterpret_tensor(buf217, (384, ), (1, ), 0), buf222, buf223, reinterpret_tensor(buf206, (384, ), (1, ), 0), buf215, buf216, reinterpret_tensor(buf198, (384, ), (1, ), 0), buf203, buf204, reinterpret_tensor(buf187, (384, ), (1, ), 0), buf196, buf197, reinterpret_tensor(buf179, (384, ), (1, ), 0), buf184, buf185, reinterpret_tensor(buf168, (384, ), (1, ), 0), buf177, buf178, reinterpret_tensor(buf160, (384, ), (1, ), 0), buf165, buf166, reinterpret_tensor(buf149, (384, ), (1, ), 0), buf158, buf159, reinterpret_tensor(buf141, (384, ), (1, ), 0), buf146, buf147, reinterpret_tensor(buf130, (384, ), (1, ), 0), buf139, buf140, reinterpret_tensor(buf122, (384, ), (1, ), 0), buf127, buf128, reinterpret_tensor(buf111, (384, ), (1, ), 0), buf120, buf121, reinterpret_tensor(buf103, (384, ), (1, ), 0), buf108, buf109, reinterpret_tensor(buf92, (384, ), (1, ), 0), buf101, buf102, reinterpret_tensor(buf84, (384, ), (1, ), 0), buf89, buf90, reinterpret_tensor(buf73, (384, ), (1, ), 0), buf82, buf83, reinterpret_tensor(buf65, (384, ), (1, ), 0), buf70, buf71, reinterpret_tensor(buf54, (384, ), (1, ), 0), buf63, buf64, reinterpret_tensor(buf46, (384, ), (1, ), 0), buf51, buf52, reinterpret_tensor(buf35, (384, ), (1, ), 0), buf44, buf45, reinterpret_tensor(buf27, (384, ), (1, ), 0), buf32, buf33, reinterpret_tensor(buf16, (384, ), (1, ), 0), buf25, buf26, buf14, buf15, buf245, buf246, reinterpret_tensor(buf239, (196, 196), (196, 1), 0), reinterpret_tensor(buf240, (196, ), (1, ), 0), reinterpret_tensor(buf232, (1536, 384), (384, 1), 0), reinterpret_tensor(buf231, (1536, ), (1, ), 0), reinterpret_tensor(buf228, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf229, (384, ), (1, ), 0), reinterpret_tensor(buf220, (196, 196), (196, 1), 0), reinterpret_tensor(buf221, (196, ), (1, ), 0), reinterpret_tensor(buf213, (1536, 384), (384, 1), 0), reinterpret_tensor(buf212, (1536, ), (1, ), 0), reinterpret_tensor(buf209, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf210, (384, ), (1, ), 0), reinterpret_tensor(buf201, (196, 196), (196, 1), 0), reinterpret_tensor(buf202, (196, ), (1, ), 0), reinterpret_tensor(buf194, (1536, 384), (384, 1), 0), reinterpret_tensor(buf193, (1536, ), (1, ), 0), reinterpret_tensor(buf190, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf191, (384, ), (1, ), 0), reinterpret_tensor(buf182, (196, 196), (196, 1), 0), reinterpret_tensor(buf183, (196, ), (1, ), 0), reinterpret_tensor(buf175, (1536, 384), (384, 1), 0), reinterpret_tensor(buf174, (1536, ), (1, ), 0), reinterpret_tensor(buf171, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf172, (384, ), (1, ), 0), reinterpret_tensor(buf163, (196, 196), (196, 1), 0), reinterpret_tensor(buf164, (196, ), (1, ), 0), reinterpret_tensor(buf156, (1536, 384), (384, 1), 0), reinterpret_tensor(buf155, (1536, ), (1, ), 0), reinterpret_tensor(buf152, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf153, (384, ), (1, ), 0), reinterpret_tensor(buf144, (196, 196), (196, 1), 0), reinterpret_tensor(buf145, (196, ), (1, ), 0), reinterpret_tensor(buf137, (1536, 384), (384, 1), 0), reinterpret_tensor(buf136, (1536, ), (1, ), 0), reinterpret_tensor(buf133, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf134, (384, ), (1, ), 0), reinterpret_tensor(buf125, (196, 196), (196, 1), 0), reinterpret_tensor(buf126, (196, ), (1, ), 0), reinterpret_tensor(buf118, (1536, 384), (384, 1), 0), reinterpret_tensor(buf117, (1536, ), (1, ), 0), reinterpret_tensor(buf114, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf115, (384, ), (1, ), 0), reinterpret_tensor(buf106, (196, 196), (196, 1), 0), reinterpret_tensor(buf107, (196, ), (1, ), 0), reinterpret_tensor(buf99, (1536, 384), (384, 1), 0), reinterpret_tensor(buf98, (1536, ), (1, ), 0), reinterpret_tensor(buf95, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf96, (384, ), (1, ), 0), reinterpret_tensor(buf87, (196, 196), (196, 1), 0), reinterpret_tensor(buf88, (196, ), (1, ), 0), reinterpret_tensor(buf80, (1536, 384), (384, 1), 0), reinterpret_tensor(buf79, (1536, ), (1, ), 0), reinterpret_tensor(buf76, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf77, (384, ), (1, ), 0), reinterpret_tensor(buf68, (196, 196), (196, 1), 0), reinterpret_tensor(buf69, (196, ), (1, ), 0), reinterpret_tensor(buf61, (1536, 384), (384, 1), 0), reinterpret_tensor(buf60, (1536, ), (1, ), 0), reinterpret_tensor(buf57, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf58, (384, ), (1, ), 0), reinterpret_tensor(buf49, (196, 196), (196, 1), 0), reinterpret_tensor(buf50, (196, ), (1, ), 0), reinterpret_tensor(buf42, (1536, 384), (384, 1), 0), reinterpret_tensor(buf41, (1536, ), (1, ), 0), reinterpret_tensor(buf38, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf39, (384, ), (1, ), 0), reinterpret_tensor(buf30, (196, 196), (196, 1), 0), reinterpret_tensor(buf31, (196, ), (1, ), 0), reinterpret_tensor(buf23, (1536, 384), (384, 1), 0), reinterpret_tensor(buf22, (1536, ), (1, ), 0), reinterpret_tensor(buf19, (384, 1536), (1536, 1), 0), reinterpret_tensor(buf20, (384, ), (1, ), 0), reinterpret_tensor(buf12, (1000, 384), (384, 1), 0), reinterpret_tensor(buf13, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((384, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_3 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_5 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_9 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_1 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_11 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_3 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_13 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_15 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_2 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_5 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_3 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_25 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_8 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_27 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_4 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_29 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_9 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_33 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_5 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_11 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_12 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_39 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_6 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_41 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_13 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_43 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_7 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_15 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_49 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_51 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_8 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_53 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_17 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_57 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_9 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_59 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_61 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_20 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_63 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_10 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_21 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_11 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    addmm_23 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    clone_36 = rand_strided((8, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_62 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_66 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_72 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_75 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_80 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_86 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_89 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_94 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_100 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_103 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_108 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_114 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_117 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_122 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_128 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_131 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_136 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_145 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_156 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_159 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_164 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_170 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_173 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_178 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_184 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_187 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_192 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_198 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_201 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_206 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_215 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    permute_220 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_226 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_229 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_74, primals_75, primals_80, primals_86, primals_92, primals_98, primals_104, primals_110, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_151, convolution, view_1, addmm, view_3, mm, view_5, addmm_1, view_7, addmm_2, view_9, mm_1, view_11, addmm_3, view_13, addmm_4, view_15, mm_2, view_17, addmm_5, view_19, addmm_6, view_21, mm_3, view_23, addmm_7, view_25, addmm_8, view_27, mm_4, view_29, addmm_9, view_31, addmm_10, view_33, mm_5, view_35, addmm_11, view_37, addmm_12, view_39, mm_6, view_41, addmm_13, view_43, addmm_14, view_45, mm_7, view_47, addmm_15, view_49, addmm_16, view_51, mm_8, view_53, addmm_17, view_55, addmm_18, view_57, mm_9, view_59, addmm_19, view_61, addmm_20, view_63, mm_10, view_65, addmm_21, view_67, addmm_22, view_69, mm_11, view_71, addmm_23, clone_36, permute_62, permute_66, permute_72, permute_75, permute_80, permute_86, permute_89, permute_94, permute_100, permute_103, permute_108, permute_114, permute_117, permute_122, permute_128, permute_131, permute_136, permute_142, permute_145, permute_150, permute_156, permute_159, permute_164, permute_170, permute_173, permute_178, permute_184, permute_187, permute_192, permute_198, permute_201, permute_206, permute_212, permute_215, permute_220, permute_226, permute_229, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resmlp_12_224', benchmark_compiled_module)
