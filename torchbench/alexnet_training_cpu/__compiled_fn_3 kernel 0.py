
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
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(121L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (121L*x1) + (363L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (363L*x0))] = tmp0;
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (25L*x1) + (25L*x1_inner) + (1600L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (64L*x2) + (1600L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(24L); x2<static_cast<long>(25L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (25L*x1) + (25L*x1_inner) + (1600L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (1600L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr5[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_max_pool2d_with_indices_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       long* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(774400L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(27L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(27L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (7040L*x1) + (193600L*x0))];
                            auto tmp1 = in_out_ptr0[static_cast<long>(64L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0))];
                            auto tmp3 = in_out_ptr0[static_cast<long>(128L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(3520L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0))];
                            auto tmp7 = in_out_ptr0[static_cast<long>(3584L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0))];
                            auto tmp9 = in_out_ptr0[static_cast<long>(3648L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0))];
                            auto tmp11 = in_out_ptr0[static_cast<long>(7040L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(7104L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0))];
                            auto tmp15 = in_out_ptr0[static_cast<long>(7168L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0))];
                            auto tmp2 = max_propagate_nan(tmp1, tmp0);
                            auto tmp4 = max_propagate_nan(tmp3, tmp2);
                            auto tmp6 = max_propagate_nan(tmp5, tmp4);
                            auto tmp8 = max_propagate_nan(tmp7, tmp6);
                            auto tmp10 = max_propagate_nan(tmp9, tmp8);
                            auto tmp12 = max_propagate_nan(tmp11, tmp10);
                            auto tmp14 = max_propagate_nan(tmp13, tmp12);
                            auto tmp16 = max_propagate_nan(tmp15, tmp14);
                            auto tmp17 = tmp1 > tmp0;
                            auto tmp18 = c10::convert<long>(1L + (2L*x2) + (110L*x1));
                            auto tmp19 = c10::convert<long>((2L*x2) + (110L*x1));
                            auto tmp20 = tmp17 ? tmp18 : tmp19;
                            auto tmp21 = tmp3 > tmp2;
                            auto tmp22 = c10::convert<long>(2L + (2L*x2) + (110L*x1));
                            auto tmp23 = tmp21 ? tmp22 : tmp20;
                            auto tmp24 = tmp5 > tmp4;
                            auto tmp25 = c10::convert<long>(55L + (2L*x2) + (110L*x1));
                            auto tmp26 = tmp24 ? tmp25 : tmp23;
                            auto tmp27 = tmp7 > tmp6;
                            auto tmp28 = c10::convert<long>(56L + (2L*x2) + (110L*x1));
                            auto tmp29 = tmp27 ? tmp28 : tmp26;
                            auto tmp30 = tmp9 > tmp8;
                            auto tmp31 = c10::convert<long>(57L + (2L*x2) + (110L*x1));
                            auto tmp32 = tmp30 ? tmp31 : tmp29;
                            auto tmp33 = tmp11 > tmp10;
                            auto tmp34 = c10::convert<long>(110L + (2L*x2) + (110L*x1));
                            auto tmp35 = tmp33 ? tmp34 : tmp32;
                            auto tmp36 = tmp13 > tmp12;
                            auto tmp37 = c10::convert<long>(111L + (2L*x2) + (110L*x1));
                            auto tmp38 = tmp36 ? tmp37 : tmp35;
                            auto tmp39 = tmp15 > tmp14;
                            auto tmp40 = c10::convert<long>(112L + (2L*x2) + (110L*x1));
                            auto tmp41 = tmp39 ? tmp40 : tmp38;
                            out_ptr0[static_cast<long>(x3 + (64L*x2) + (1728L*x1) + (46656L*x0))] = tmp16;
                            out_ptr1[static_cast<long>(x3 + (64L*x2) + (1728L*x1) + (46656L*x0))] = tmp41;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_max_pool2d_with_indices_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       long* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(559872L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(13L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(13L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (10368L*x1) + (139968L*x0))];
                            auto tmp1 = in_out_ptr0[static_cast<long>(192L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0))];
                            auto tmp3 = in_out_ptr0[static_cast<long>(384L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(5184L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0))];
                            auto tmp7 = in_out_ptr0[static_cast<long>(5376L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0))];
                            auto tmp9 = in_out_ptr0[static_cast<long>(5568L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0))];
                            auto tmp11 = in_out_ptr0[static_cast<long>(10368L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(10560L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0))];
                            auto tmp15 = in_out_ptr0[static_cast<long>(10752L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0))];
                            auto tmp2 = max_propagate_nan(tmp1, tmp0);
                            auto tmp4 = max_propagate_nan(tmp3, tmp2);
                            auto tmp6 = max_propagate_nan(tmp5, tmp4);
                            auto tmp8 = max_propagate_nan(tmp7, tmp6);
                            auto tmp10 = max_propagate_nan(tmp9, tmp8);
                            auto tmp12 = max_propagate_nan(tmp11, tmp10);
                            auto tmp14 = max_propagate_nan(tmp13, tmp12);
                            auto tmp16 = max_propagate_nan(tmp15, tmp14);
                            auto tmp17 = tmp1 > tmp0;
                            auto tmp18 = c10::convert<long>(1L + (2L*x2) + (54L*x1));
                            auto tmp19 = c10::convert<long>((2L*x2) + (54L*x1));
                            auto tmp20 = tmp17 ? tmp18 : tmp19;
                            auto tmp21 = tmp3 > tmp2;
                            auto tmp22 = c10::convert<long>(2L + (2L*x2) + (54L*x1));
                            auto tmp23 = tmp21 ? tmp22 : tmp20;
                            auto tmp24 = tmp5 > tmp4;
                            auto tmp25 = c10::convert<long>(27L + (2L*x2) + (54L*x1));
                            auto tmp26 = tmp24 ? tmp25 : tmp23;
                            auto tmp27 = tmp7 > tmp6;
                            auto tmp28 = c10::convert<long>(28L + (2L*x2) + (54L*x1));
                            auto tmp29 = tmp27 ? tmp28 : tmp26;
                            auto tmp30 = tmp9 > tmp8;
                            auto tmp31 = c10::convert<long>(29L + (2L*x2) + (54L*x1));
                            auto tmp32 = tmp30 ? tmp31 : tmp29;
                            auto tmp33 = tmp11 > tmp10;
                            auto tmp34 = c10::convert<long>(54L + (2L*x2) + (54L*x1));
                            auto tmp35 = tmp33 ? tmp34 : tmp32;
                            auto tmp36 = tmp13 > tmp12;
                            auto tmp37 = c10::convert<long>(55L + (2L*x2) + (54L*x1));
                            auto tmp38 = tmp36 ? tmp37 : tmp35;
                            auto tmp39 = tmp15 > tmp14;
                            auto tmp40 = c10::convert<long>(56L + (2L*x2) + (54L*x1));
                            auto tmp41 = tmp39 ? tmp40 : tmp38;
                            out_ptr0[static_cast<long>(x3 + (192L*x2) + (2496L*x1) + (32448L*x0))] = tmp16;
                            out_ptr1[static_cast<long>(x3 + (192L*x2) + (2496L*x1) + (32448L*x0))] = tmp41;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(259584L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_relu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(173056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__adaptive_avg_pool2d_max_pool2d_with_indices_relu_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       long* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(173056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (6656L*x1) + (43264L*x0))];
                                auto tmp1 = in_out_ptr0[static_cast<long>(256L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0))];
                                auto tmp3 = in_out_ptr0[static_cast<long>(512L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0))];
                                auto tmp5 = in_out_ptr0[static_cast<long>(3328L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0))];
                                auto tmp7 = in_out_ptr0[static_cast<long>(3584L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0))];
                                auto tmp9 = in_out_ptr0[static_cast<long>(3840L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0))];
                                auto tmp11 = in_out_ptr0[static_cast<long>(6656L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0))];
                                auto tmp13 = in_out_ptr0[static_cast<long>(6912L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0))];
                                auto tmp15 = in_out_ptr0[static_cast<long>(7168L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0))];
                                auto tmp2 = max_propagate_nan(tmp1, tmp0);
                                auto tmp4 = max_propagate_nan(tmp3, tmp2);
                                auto tmp6 = max_propagate_nan(tmp5, tmp4);
                                auto tmp8 = max_propagate_nan(tmp7, tmp6);
                                auto tmp10 = max_propagate_nan(tmp9, tmp8);
                                auto tmp12 = max_propagate_nan(tmp11, tmp10);
                                auto tmp14 = max_propagate_nan(tmp13, tmp12);
                                auto tmp16 = max_propagate_nan(tmp15, tmp14);
                                auto tmp17 = tmp1 > tmp0;
                                auto tmp18 = c10::convert<long>(1L + (2L*x2) + (26L*x1));
                                auto tmp19 = c10::convert<long>((2L*x2) + (26L*x1));
                                auto tmp20 = tmp17 ? tmp18 : tmp19;
                                auto tmp21 = tmp3 > tmp2;
                                auto tmp22 = c10::convert<long>(2L + (2L*x2) + (26L*x1));
                                auto tmp23 = tmp21 ? tmp22 : tmp20;
                                auto tmp24 = tmp5 > tmp4;
                                auto tmp25 = c10::convert<long>(13L + (2L*x2) + (26L*x1));
                                auto tmp26 = tmp24 ? tmp25 : tmp23;
                                auto tmp27 = tmp7 > tmp6;
                                auto tmp28 = c10::convert<long>(14L + (2L*x2) + (26L*x1));
                                auto tmp29 = tmp27 ? tmp28 : tmp26;
                                auto tmp30 = tmp9 > tmp8;
                                auto tmp31 = c10::convert<long>(15L + (2L*x2) + (26L*x1));
                                auto tmp32 = tmp30 ? tmp31 : tmp29;
                                auto tmp33 = tmp11 > tmp10;
                                auto tmp34 = c10::convert<long>(26L + (2L*x2) + (26L*x1));
                                auto tmp35 = tmp33 ? tmp34 : tmp32;
                                auto tmp36 = tmp13 > tmp12;
                                auto tmp37 = c10::convert<long>(27L + (2L*x2) + (26L*x1));
                                auto tmp38 = tmp36 ? tmp37 : tmp35;
                                auto tmp39 = tmp15 > tmp14;
                                auto tmp40 = c10::convert<long>(28L + (2L*x2) + (26L*x1));
                                auto tmp41 = tmp39 ? tmp40 : tmp38;
                                out_ptr0[static_cast<long>(x3 + (256L*x2) + (1536L*x1) + (9216L*x0))] = tmp16;
                                out_ptr1[static_cast<long>(x3 + (256L*x2) + (1536L*x1) + (9216L*x0))] = tmp41;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9216L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((256L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(36L))) + (9216L*x0) + (c10::div_floor_integer((x1 + x1_inner), 36L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (9216L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = tmp0 * (tmp0>0);
            auto tmp2 = static_cast<float>(0.0);
            auto tmp3 = tmp1 <= tmp2;
            in_out_ptr0[static_cast<long>(x0)] = tmp1;
            out_ptr0[static_cast<long>(x0)] = tmp3;
        }
    }
}
''')


cpp_fused_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 11, 11), (363, 121, 11, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (192, 64, 5, 5), (1600, 25, 5, 1))
    assert_size_stride(primals_4, (192, ), (1, ))
    assert_size_stride(primals_5, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_7, (256, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (4096, 9216), (9216, 1))
    assert_size_stride(primals_12, (4096, ), (1, ))
    assert_size_stride(primals_13, (4096, 4096), (4096, 1))
    assert_size_stride(primals_14, (4096, ), (1, ))
    assert_size_stride(primals_15, (1000, 4096), (4096, 1))
    assert_size_stride(primals_16, (1000, ), (1, ))
    assert_size_stride(primals_17, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((64, 3, 11, 11), (363, 1, 33, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((192, 64, 5, 5), (1600, 1, 320, 64), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((256, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del primals_1
    del primals_17
    del primals_3
    del primals_5
    del primals_7
    del primals_9
    # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
    buf6 = extern_kernels.convolution(buf5, buf0, primals_2, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf6, (4, 64, 55, 55), (193600, 1, 3520, 64))
    del primals_2
    buf7 = buf6; del buf6  # reuse
    buf8 = empty_strided((4, 64, 27, 27), (46656, 1, 1728, 64), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((4, 64, 27, 27), (46656, 1, 1728, 64), device='cpu', dtype=torch.int64)
    cpp_fused_max_pool2d_with_indices_relu_1(c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    # Source Nodes: [l__mod___features_3], Original ATen: [aten.convolution]
    buf10 = extern_kernels.convolution(buf8, buf1, primals_4, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf10, (4, 192, 27, 27), (139968, 1, 5184, 192))
    del primals_4
    buf11 = buf10; del buf10  # reuse
    buf12 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.int64)
    cpp_fused_max_pool2d_with_indices_relu_2(c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    # Source Nodes: [l__mod___features_6], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf12, buf2, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf14, (4, 384, 13, 13), (64896, 1, 4992, 384))
    del primals_6
    buf15 = buf14; del buf14  # reuse
    cpp_fused_relu_3(c_void_p(buf15.data_ptr()))
    # Source Nodes: [l__mod___features_8], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf15, buf3, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf16, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del primals_8
    buf17 = buf16; del buf16  # reuse
    cpp_fused_relu_4(c_void_p(buf17.data_ptr()))
    # Source Nodes: [l__mod___features_10], Original ATen: [aten.convolution]
    buf18 = extern_kernels.convolution(buf17, buf4, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf18, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del primals_10
    buf19 = buf18; del buf18  # reuse
    buf20 = empty_strided((4, 256, 6, 6), (9216, 1, 1536, 256), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((4, 256, 6, 6), (9216, 1, 1536, 256), device='cpu', dtype=torch.int64)
    buf22 = empty((4, 9216), device='cpu', dtype=torch.float32)
    cpp_fused__adaptive_avg_pool2d_max_pool2d_with_indices_relu_view_5(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    buf23 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_12, buf22, reinterpret_tensor(primals_11, (9216, 4096), (1, 9216), 0), alpha=1, beta=1, out=buf23)
    del primals_12
    buf24 = buf23; del buf23  # reuse
    buf28 = empty((4, 4096), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_6(c_void_p(buf24.data_ptr()), c_void_p(buf28.data_ptr()))
    buf25 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_14, buf24, reinterpret_tensor(primals_13, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf25)
    del primals_14
    buf26 = buf25; del buf25  # reuse
    cpp_fused_relu_7(c_void_p(buf26.data_ptr()))
    buf27 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_16, buf26, reinterpret_tensor(primals_15, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf27)
    del primals_16
    return (buf27, buf0, buf1, buf2, buf3, buf4, buf5, buf7, buf8, buf9, buf11, buf12, buf13, buf15, buf17, buf19, buf20, buf21, buf22, buf24, buf26, reinterpret_tensor(primals_15, (1000, 4096), (4096, 1), 0), reinterpret_tensor(primals_13, (4096, 4096), (4096, 1), 0), buf28, reinterpret_tensor(primals_11, (4096, 9216), (9216, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 11, 11), (363, 121, 11, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((192, 64, 5, 5), (1600, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((256, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((4096, 9216), (9216, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((1000, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('alexnet', benchmark_compiled_module)
